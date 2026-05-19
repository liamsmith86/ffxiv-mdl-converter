// SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use byteorder::{LittleEndian, WriteBytesExt};
use half::f16;
use rusqlite::Connection;

use crate::mdl::{
    MeshLod, ModelFile, VertexDeclaration as MdlVertexDeclaration, VertexType as MdlVertexType,
    VertexUsage as MdlVertexUsage, parse_mdl,
};

const MDL_HEADER_SIZE: usize = 68;
const VERTEX_DECLARATION_SLOTS: usize = 17;
const VERTEX_DECLARATION_BYTES: usize = VERTEX_DECLARATION_SLOTS * 8;
const MAX_VERTEX_BUFFER_SIZE: usize = 0xFFFF_FFFFusize;
const EXTRA_LOD_ENABLED: u8 = 0x10;
const BONELESS_PARTS_ENABLED: u8 = 0x20;

#[derive(Debug, Clone)]
pub struct WrapOptions {
    pub base_mdl: Option<PathBuf>,
    pub copy_materials: bool,
    pub copy_attributes: bool,
    pub autoscale: bool,
    pub shift_import_uv: bool,
    pub clone_uv2: bool,
    pub use_imported_tangents: bool,
    pub auto_assign_heels: bool,
}

impl Default for WrapOptions {
    fn default() -> Self {
        Self {
            base_mdl: None,
            copy_materials: true,
            copy_attributes: true,
            autoscale: true,
            shift_import_uv: true,
            clone_uv2: false,
            use_imported_tangents: false,
            auto_assign_heels: true,
        }
    }
}

#[derive(Debug, Clone)]
struct ImportModel {
    materials: Vec<String>,
    meshes: Vec<ImportMesh>,
    attributes: Vec<String>,
    bones: Vec<String>,
    shape_names: Vec<String>,
    shape_parts: Vec<CompiledShapePart>,
    shape_value_count: usize,
    usage: UsageInfo,
}

#[derive(Debug, Clone)]
struct ImportMesh {
    mesh_id: i64,
    material_id: usize,
    mesh_type: MeshType,
    bones: Vec<String>,
    bone_global_indices: Vec<usize>,
    parts: Vec<ImportPart>,
    base_vertex_count: usize,
    total_vertex_count: usize,
    index_count: usize,
}

#[derive(Debug, Clone)]
struct ImportPart {
    part_id: i64,
    attributes: Vec<String>,
    vertices: Vec<ImportVertex>,
    indices: Vec<u32>,
    shape_vertices: BTreeMap<String, Vec<ShapeVertex>>,
    vertex_start: usize,
}

#[derive(Debug, Clone)]
struct ShapeVertex {
    vertex_id: usize,
    position: [f32; 3],
}

#[derive(Debug, Clone)]
struct CompiledShapePart {
    shape_name: String,
    mesh_index: usize,
    replacements: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MeshType {
    Standard,
    Water,
    Shadow,
    TerrainShadow,
    Fog,
    LightShaft,
    Glass,
    MaterialChange,
    CrestChange,
    Unknown,
}

impl MeshType {
    fn from_db(value: &str) -> Self {
        match value.to_ascii_lowercase().as_str() {
            "standard" | "" => Self::Standard,
            "water" => Self::Water,
            "shadow" => Self::Shadow,
            "terrainshadow" | "terrain_shadow" | "terrain shadow" => Self::TerrainShadow,
            "fog" => Self::Fog,
            "lightshaft" | "light_shaft" | "light shaft" => Self::LightShaft,
            "glass" => Self::Glass,
            "materialchange" | "material_change" | "material change" => Self::MaterialChange,
            "crestchange" | "crest_change" | "crest change" => Self::CrestChange,
            _ => Self::Unknown,
        }
    }

    fn from_lod_mesh_index(lod: &MeshLod, mesh_index: usize) -> Self {
        let mesh_index = mesh_index as u16;
        if in_lod_range(mesh_index, lod.water_mesh_index, lod.water_mesh_count) {
            Self::Water
        } else if in_lod_range(mesh_index, lod.shadow_mesh_index, lod.shadow_mesh_count) {
            Self::Shadow
        } else if in_lod_range(
            mesh_index,
            lod.terrain_shadow_mesh_index,
            lod.terrain_shadow_mesh_count,
        ) {
            Self::TerrainShadow
        } else if in_lod_range(
            mesh_index,
            lod.vertical_fog_mesh_index,
            lod.vertical_fog_mesh_count,
        ) {
            Self::Fog
        } else {
            Self::Standard
        }
    }
}

fn in_lod_range(mesh_index: u16, range_start: u16, range_count: u16) -> bool {
    range_count > 0 && mesh_index >= range_start && mesh_index < range_start + range_count
}

#[derive(Debug, Clone, Copy, Default)]
struct UsageInfo {
    has_weights: bool,
    needs_eight_weights: bool,
    max_uv: u8,
    uses_color2: bool,
    uses_flow: bool,
}

#[derive(Debug, Clone, Copy)]
struct ImportVertex {
    position: [f32; 3],
    normal: [f32; 3],
    binormal: [f32; 3],
    tangent: [f32; 3],
    color: [u8; 4],
    color2: [u8; 4],
    uv1: [f32; 2],
    uv2: [f32; 2],
    uv3: [f32; 2],
    bone_ids: [u8; 8],
    bone_weights: [u8; 8],
    flow: [f32; 3],
}

#[derive(Debug, Default)]
struct GeometryData {
    vertex_data0: Vec<u8>,
    vertex_data1: Vec<u8>,
    vertex_data2: Vec<u8>,
    index_data: Vec<u8>,
}

#[derive(Debug, Clone)]
struct VertexLayout {
    elements: Vec<VertexElement>,
    stream_strides: [u8; 3],
    stream_count: u8,
}

#[derive(Debug, Clone, Copy)]
struct VertexElement {
    stream: u8,
    offset: u8,
    data_type: VertexDataType,
    usage: VertexUsage,
    usage_index: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum VertexDataType {
    Float1 = 0x00,
    Float2 = 0x01,
    Float3 = 0x02,
    Float4 = 0x03,
    Ubyte4 = 0x05,
    Short2 = 0x06,
    Short4 = 0x07,
    Ubyte4n = 0x08,
    Short2n = 0x09,
    Short4n = 0x0A,
    Ushort2n = 0x0B,
    Ushort4n = 0x0C,
    Half2 = 0x0D,
    Half4 = 0x0E,
    Ubyte8 = 0x11,
}

impl VertexDataType {
    fn from_mdl(value: MdlVertexType) -> Self {
        match value {
            MdlVertexType::Float1 => Self::Float1,
            MdlVertexType::Float2 => Self::Float2,
            MdlVertexType::Float3 => Self::Float3,
            MdlVertexType::Float4 => Self::Float4,
            MdlVertexType::Ubyte4 => Self::Ubyte4,
            MdlVertexType::Short2 => Self::Short2,
            MdlVertexType::Short4 => Self::Short4,
            MdlVertexType::Ubyte4n => Self::Ubyte4n,
            MdlVertexType::Short2n => Self::Short2n,
            MdlVertexType::Short4n => Self::Short4n,
            MdlVertexType::Ushort2n => Self::Ushort2n,
            MdlVertexType::Ushort4n => Self::Ushort4n,
            MdlVertexType::Half2 => Self::Half2,
            MdlVertexType::Half4 => Self::Half4,
            MdlVertexType::Ubyte8 => Self::Ubyte8,
        }
    }

    fn size(self) -> u8 {
        match self {
            Self::Float1 => 4,
            Self::Float2 => 8,
            Self::Float3 => 12,
            Self::Float4 => 16,
            Self::Short2 | Self::Short2n | Self::Ushort2n | Self::Half2 => 4,
            Self::Short4 | Self::Short4n | Self::Ushort4n | Self::Half4 => 8,
            Self::Ubyte4 | Self::Ubyte4n => 4,
            Self::Ubyte8 => 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum VertexUsage {
    Position = 0,
    BoneWeight = 1,
    BoneIndex = 2,
    Normal = 3,
    TextureCoordinate = 4,
    Flow = 5,
    Binormal = 6,
    Color = 7,
}

impl VertexUsage {
    fn from_mdl(value: MdlVertexUsage) -> Self {
        match value {
            MdlVertexUsage::Position => Self::Position,
            MdlVertexUsage::BoneWeight => Self::BoneWeight,
            MdlVertexUsage::BoneIndex => Self::BoneIndex,
            MdlVertexUsage::Normal => Self::Normal,
            MdlVertexUsage::TextureCoordinate => Self::TextureCoordinate,
            MdlVertexUsage::Flow => Self::Flow,
            MdlVertexUsage::Binormal => Self::Binormal,
            MdlVertexUsage::Color => Self::Color,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct Bounds {
    min: [f32; 3],
    max: [f32; 3],
    radius: f32,
}

#[derive(Debug, Clone)]
struct MeshGeometrySummary {
    vertex_offsets: [u32; 3],
    index_offset: u32,
    total_vertex_count: usize,
    index_count: usize,
}

#[derive(Debug, Clone)]
struct MeshPartSummary {
    index_offset: u32,
    index_count: u32,
    attribute_mask: u32,
    bone_start: i16,
    bone_count: u16,
}

#[derive(Debug, Clone)]
struct PreservedBlocks {
    element_ids: Vec<u8>,
    terrain_shadow_meshes: Vec<u8>,
    terrain_shadow_submeshes: Vec<u8>,
    neck_morph: Vec<u8>,
    patch72: Vec<u8>,
    padding: Vec<u8>,
}

pub fn wrap_db_to_mdl(
    source_db: &Path,
    dest_mdl: &Path,
    game_path: &str,
    options: &WrapOptions,
) -> Result<()> {
    let base_path = resolve_base_mdl(game_path, options.base_mdl.as_deref())?;
    let base = parse_mdl(&base_path)
        .with_context(|| format!("failed to parse base MDL {}", base_path.display()))?;
    let mut model = load_import_db(source_db, options.shift_import_uv, options.clone_uv2)
        .with_context(|| format!("failed to read DB {}", source_db.display()))?;
    apply_wrap_options(&mut model, &base, game_path, options)?;
    compile_model_metadata(&mut model)?;
    let bytes = make_uncompressed_mdl(&model, &base, options)?;

    if let Some(parent) = dest_mdl.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let temp_path = temp_mdl_path(dest_mdl);
    fs::write(&temp_path, bytes)
        .with_context(|| format!("failed to write {}", temp_path.display()))?;
    if dest_mdl.exists() {
        fs::remove_file(dest_mdl)?;
    }
    fs::rename(&temp_path, dest_mdl).with_context(|| {
        format!(
            "failed to move {} to {}",
            temp_path.display(),
            dest_mdl.display()
        )
    })?;
    Ok(())
}

fn resolve_base_mdl(game_path: &str, explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return Ok(path.to_path_buf());
    }
    if Path::new(game_path).is_file() {
        return Ok(PathBuf::from(game_path));
    }

    let normalized_game_path = game_path.trim_start_matches(['/', '\\']).replace('\\', "/");
    for root in base_search_roots() {
        let candidate = root.join(&normalized_game_path);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    bail!(
        "Rust wrap could not resolve a base MDL for '{game_path}'. Pass --base-mdl or set FFXIV_MDL_BASE_DIRS/FFXIV_MDL_BASE_DIR/FFXIV_GAMEFILES_EXTRACTED to an extracted game-file mirror."
    )
}

fn base_search_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for key in [
        "FFXIV_MDL_BASE_DIRS",
        "FFXIV_MDL_BASE_DIR",
        "FFXIV_GAMEFILES_EXTRACTED",
        "FFXIV_GAMEFILES_DIR",
    ] {
        let Ok(value) = env::var(key) else {
            continue;
        };
        roots.extend(
            env::split_paths(&value)
                .filter(|path| !path.as_os_str().is_empty())
                .flat_map(|path| {
                    [
                        path.clone(),
                        path.join("extracted"),
                        path.join("ffxiv-gamefiles").join("extracted"),
                    ]
                }),
        );
    }
    roots
}

fn load_import_db(path: &Path, shift_import_uv: bool, clone_uv2: bool) -> Result<ImportModel> {
    let conn = Connection::open(path)?;
    migrate_optional_columns(&conn)?;

    let _root_name = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'root_name'",
            [],
            |row| row.get::<_, String>(0),
        )
        .unwrap_or_else(|_| "model".to_string());

    let materials = load_materials(&conn)?;
    let mut meshes = load_meshes(&conn, &materials)?;
    for mesh in &mut meshes {
        mesh.bones = load_bones(&conn, mesh.mesh_id)?;
        mesh.parts = load_parts(&conn, mesh.mesh_id)?;
        for part in &mut mesh.parts {
            part.vertices = load_vertices(
                &conn,
                mesh.mesh_id,
                part.part_id,
                shift_import_uv,
                clone_uv2,
            )?;
            part.indices = load_indices(&conn, mesh.mesh_id, part.part_id)?;
            part.shape_vertices = load_shape_vertices(&conn, mesh.mesh_id, part.part_id)?;
        }
    }

    Ok(ImportModel {
        materials,
        meshes,
        attributes: Vec::new(),
        bones: Vec::new(),
        shape_names: Vec::new(),
        shape_parts: Vec::new(),
        shape_value_count: 0,
        usage: UsageInfo::default(),
    })
}

fn apply_wrap_options(
    model: &mut ImportModel,
    base: &ModelFile,
    game_path: &str,
    options: &WrapOptions,
) -> Result<()> {
    if options.copy_materials {
        copy_base_materials(model, base);
    }
    if options.copy_attributes {
        copy_base_attributes(model, base);
    }
    if options.autoscale {
        auto_scale_model(model, base);
    }
    if options.auto_assign_heels {
        assign_heel_attribute(model, game_path);
    }
    Ok(())
}

fn copy_base_materials(model: &mut ImportModel, base: &ModelFile) {
    let original_materials = model.materials.clone();
    let mut copied_materials = Vec::<String>::new();
    let mut material_ids = HashMap::<String, usize>::new();

    for (mesh_index, mesh) in model.meshes.iter_mut().enumerate() {
        let material = base
            .lod0_meshes
            .get(mesh_index)
            .map(|base_mesh| base.data.material_name(base_mesh.mesh.material_index))
            .filter(|name| !name.is_empty())
            .or_else(|| original_materials.get(mesh.material_id).cloned())
            .unwrap_or_else(|| "/mt_c0001b0001_a.mtrl".to_string());
        mesh.material_id =
            get_or_insert_material(&mut copied_materials, &mut material_ids, &material);
    }

    model.materials = copied_materials;
}

fn get_or_insert_material(
    materials: &mut Vec<String>,
    material_ids: &mut HashMap<String, usize>,
    material: &str,
) -> usize {
    let normalized = normalize_material_path(material);
    if let Some(index) = material_ids.get(&normalized) {
        return *index;
    }
    let index = materials.len();
    materials.push(normalized.clone());
    material_ids.insert(normalized, index);
    index
}

fn copy_base_attributes(model: &mut ImportModel, base: &ModelFile) {
    let lod0 = base.data.lods.first();
    for (mesh_index, mesh) in model.meshes.iter_mut().enumerate() {
        let Some(base_mesh) = base.lod0_meshes.get(mesh_index) else {
            continue;
        };
        if let Some(lod) = lod0 {
            mesh.mesh_type = MeshType::from_lod_mesh_index(lod, base_mesh.original_mesh_index);
        }
        for (part_index, part) in mesh.parts.iter_mut().enumerate() {
            let Some(base_part) = base_mesh.submeshes.get(part_index) else {
                continue;
            };
            part.attributes = base
                .data
                .attributes_for_mask(base_part.attribute_index_mask);
        }
    }
}

const AUTOSCALE_TOLERANCE: f32 = 0.3;
const AUTOSCALE_FACTORS: &[f32] = &[
    1.0,
    10.0,
    100.0,
    1000.0,
    0.1,
    0.01,
    0.001,
    0.0001,
    0.003_937_008,
    0.039_370_08,
    0.393_700_78,
    3.937_008,
    39.370_08,
    254.0,
    25.40,
    2.540,
    0.254,
    0.0254,
    0.00254,
];

fn auto_scale_model(model: &mut ImportModel, base: &ModelFile) {
    let imported_diag = bounds_diagonal(calculate_bounds(model));
    let base_diag = base_lod0_diagonal(base);
    if imported_diag <= f32::EPSILON || base_diag <= f32::EPSILON {
        return;
    }

    for factor in AUTOSCALE_FACTORS {
        let scaled_diag = imported_diag * *factor;
        if ((base_diag - scaled_diag).abs() / base_diag) <= AUTOSCALE_TOLERANCE {
            if (*factor - 1.0).abs() > f32::EPSILON {
                scale_model(model, *factor);
            }
            return;
        }
    }
}

fn base_lod0_diagonal(base: &ModelFile) -> f32 {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let mut any = false;
    for mesh in &base.lod0_meshes {
        for vertex in &mesh.vertices {
            any = true;
            for axis in 0..3 {
                min[axis] = min[axis].min(vertex.position[axis]);
                max[axis] = max[axis].max(vertex.position[axis]);
            }
        }
    }
    if any {
        diagonal_from_min_max(min, max)
    } else {
        0.0
    }
}

fn bounds_diagonal(bounds: Bounds) -> f32 {
    diagonal_from_min_max(bounds.min, bounds.max)
}

fn diagonal_from_min_max(min: [f32; 3], max: [f32; 3]) -> f32 {
    let dx = max[0] - min[0];
    let dy = max[1] - min[1];
    let dz = max[2] - min[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn scale_model(model: &mut ImportModel, scale: f32) {
    for mesh in &mut model.meshes {
        for part in &mut mesh.parts {
            for vertex in &mut part.vertices {
                for value in &mut vertex.position {
                    *value *= scale;
                }
            }
            for shape_vertices in part.shape_vertices.values_mut() {
                for vertex in shape_vertices {
                    for value in &mut vertex.position {
                        *value *= scale;
                    }
                }
            }
        }
    }
}

fn assign_heel_attribute(model: &mut ImportModel, game_path: &str) {
    let normalized = game_path.to_ascii_lowercase();
    if !normalized.ends_with("_top.mdl")
        && !normalized.ends_with("_dwn.mdl")
        && !normalized.ends_with("_sho.mdl")
    {
        return;
    }

    let min_y = model
        .meshes
        .iter()
        .flat_map(|mesh| &mesh.parts)
        .flat_map(|part| &part.vertices)
        .map(|vertex| vertex.position[1])
        .fold(f32::INFINITY, f32::min);
    if !min_y.is_finite() || min_y >= 0.0 {
        return;
    }

    for mesh in &mut model.meshes {
        for part in &mut mesh.parts {
            part.attributes
                .retain(|attribute| !attribute.starts_with("heels_offset="));
        }
    }
    let Some(first_part) = model
        .meshes
        .first_mut()
        .and_then(|mesh| mesh.parts.first_mut())
    else {
        return;
    };
    first_part
        .attributes
        .push(format!("heels_offset={:.4}", -min_y));
}

fn migrate_optional_columns(conn: &Connection) -> Result<()> {
    let columns = table_columns(conn, "vertices")?;
    for (name, ddl) in [
        (
            "uv_3_u",
            "ALTER TABLE vertices ADD COLUMN uv_3_u REAL NOT NULL DEFAULT 0",
        ),
        (
            "uv_3_v",
            "ALTER TABLE vertices ADD COLUMN uv_3_v REAL NOT NULL DEFAULT 0",
        ),
        (
            "flow_u",
            "ALTER TABLE vertices ADD COLUMN flow_u REAL NOT NULL DEFAULT 0",
        ),
        (
            "flow_v",
            "ALTER TABLE vertices ADD COLUMN flow_v REAL NOT NULL DEFAULT 0",
        ),
    ] {
        if !columns.iter().any(|column| column == name) {
            conn.execute(ddl, [])?;
        }
    }
    Ok(())
}

fn table_columns(conn: &Connection, table: &str) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(Into::into)
}

fn load_materials(conn: &Connection) -> Result<Vec<String>> {
    let mut stmt = conn.prepare("SELECT material_id, name FROM materials ORDER BY material_id")?;
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
    })?;
    let mut by_id = BTreeMap::new();
    for row in rows {
        let (id, name) = row?;
        if id < 0 {
            bail!("material_id cannot be negative: {id}");
        }
        by_id.insert(id as usize, normalize_material_path(&name));
    }
    if by_id.is_empty() {
        bail!("DB has no materials");
    }
    let max_id = *by_id.keys().next_back().unwrap_or(&0);
    let mut materials = Vec::with_capacity(max_id + 1);
    for id in 0..=max_id {
        let material = by_id
            .get(&id)
            .with_context(|| format!("materials table is missing material_id {id}"))?;
        materials.push(material.clone());
    }
    Ok(materials)
}

fn load_meshes(conn: &Connection, materials: &[String]) -> Result<Vec<ImportMesh>> {
    let mut stmt = conn.prepare(
        "SELECT mesh, material_id, COALESCE(type, 'Standard') FROM meshes ORDER BY mesh",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, i64>(1)?,
            row.get::<_, String>(2)?,
        ))
    })?;
    let mut meshes = Vec::new();
    for row in rows {
        let (mesh_id, material_id, mesh_type) = row?;
        if material_id < 0 || material_id as usize >= materials.len() {
            bail!("mesh {mesh_id} references invalid material_id {material_id}");
        }
        meshes.push(ImportMesh {
            mesh_id,
            material_id: material_id as usize,
            mesh_type: MeshType::from_db(&mesh_type),
            bones: Vec::new(),
            bone_global_indices: Vec::new(),
            parts: Vec::new(),
            base_vertex_count: 0,
            total_vertex_count: 0,
            index_count: 0,
        });
    }
    if meshes.is_empty() {
        bail!("DB has no meshes");
    }
    Ok(meshes)
}

fn load_bones(conn: &Connection, mesh_id: i64) -> Result<Vec<String>> {
    let mut stmt =
        conn.prepare("SELECT bone_id, name FROM bones WHERE mesh = ?1 ORDER BY bone_id")?;
    let rows = stmt.query_map([mesh_id], |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
    })?;
    let mut bones = Vec::new();
    for row in rows {
        let (bone_id, name) = row?;
        if bone_id < 0 {
            bail!("mesh {mesh_id} has negative bone_id {bone_id}");
        }
        if bone_id as usize != bones.len() {
            bail!("mesh {mesh_id} bone IDs must be contiguous from 0; found {bone_id}");
        }
        bones.push(name);
    }
    Ok(bones)
}

fn load_parts(conn: &Connection, mesh_id: i64) -> Result<Vec<ImportPart>> {
    let mut stmt = conn.prepare(
        "SELECT part, COALESCE(attributes, '') FROM parts WHERE mesh = ?1 ORDER BY part",
    )?;
    let rows = stmt.query_map([mesh_id], |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
    })?;
    let mut parts = Vec::new();
    for row in rows {
        let (part_id, attributes) = row?;
        parts.push(ImportPart {
            part_id,
            attributes: split_attributes(&attributes),
            vertices: Vec::new(),
            indices: Vec::new(),
            shape_vertices: BTreeMap::new(),
            vertex_start: 0,
        });
    }
    if parts.is_empty() {
        parts.push(ImportPart {
            part_id: 0,
            attributes: Vec::new(),
            vertices: Vec::new(),
            indices: Vec::new(),
            shape_vertices: BTreeMap::new(),
            vertex_start: 0,
        });
    }
    Ok(parts)
}

fn load_vertices(
    conn: &Connection,
    mesh_id: i64,
    part_id: i64,
    shift_import_uv: bool,
    clone_uv2: bool,
) -> Result<Vec<ImportVertex>> {
    let mut stmt = conn.prepare(
        "SELECT
            position_x, position_y, position_z,
            normal_x, normal_y, normal_z,
            COALESCE(binormal_x, 0), COALESCE(binormal_y, 0), COALESCE(binormal_z, 0),
            COALESCE(tangent_x, 0), COALESCE(tangent_y, 0), COALESCE(tangent_z, 0),
            color_r, color_g, color_b, color_a,
            color2_r, color2_g, color2_b, color2_a,
            uv_1_u, uv_1_v, uv_2_u, uv_2_v, uv_3_u, uv_3_v,
            COALESCE(bone_1_id, 0), COALESCE(bone_1_weight, 0),
            COALESCE(bone_2_id, 0), COALESCE(bone_2_weight, 0),
            COALESCE(bone_3_id, 0), COALESCE(bone_3_weight, 0),
            COALESCE(bone_4_id, 0), COALESCE(bone_4_weight, 0),
            COALESCE(bone_5_id, 0), COALESCE(bone_5_weight, 0),
            COALESCE(bone_6_id, 0), COALESCE(bone_6_weight, 0),
            COALESCE(bone_7_id, 0), COALESCE(bone_7_weight, 0),
            COALESCE(bone_8_id, 0), COALESCE(bone_8_weight, 0),
            COALESCE(flow_u, 0), COALESCE(flow_v, 0)
         FROM vertices WHERE mesh = ?1 AND part = ?2 ORDER BY vertex_id",
    )?;
    let rows = stmt.query_map((mesh_id, part_id), |row| {
        let mut bone_ids = [0u8; 8];
        let mut weights = [0.0f32; 8];
        for index in 0..8 {
            let id: i64 = row.get(26 + index * 2)?;
            bone_ids[index] = id.clamp(0, u8::MAX as i64) as u8;
            weights[index] = row.get::<_, f64>(27 + index * 2)? as f32;
        }
        let uv1 = import_uv(
            [row.get::<_, f64>(20)? as f32, row.get::<_, f64>(21)? as f32],
            shift_import_uv,
        );
        let uv2 = if clone_uv2 {
            uv1
        } else {
            import_uv(
                [row.get::<_, f64>(22)? as f32, row.get::<_, f64>(23)? as f32],
                shift_import_uv,
            )
        };
        let uv3 = import_uv(
            [row.get::<_, f64>(24)? as f32, row.get::<_, f64>(25)? as f32],
            shift_import_uv,
        );
        Ok(ImportVertex {
            position: [
                row.get::<_, f64>(0)? as f32,
                row.get::<_, f64>(1)? as f32,
                row.get::<_, f64>(2)? as f32,
            ],
            normal: normalize3_or(
                [
                    row.get::<_, f64>(3)? as f32,
                    row.get::<_, f64>(4)? as f32,
                    row.get::<_, f64>(5)? as f32,
                ],
                [0.0, 1.0, 0.0],
            ),
            binormal: normalize3_or(
                [
                    row.get::<_, f64>(6)? as f32,
                    row.get::<_, f64>(7)? as f32,
                    row.get::<_, f64>(8)? as f32,
                ],
                [1.0, 0.0, 0.0],
            ),
            tangent: normalize3_or(
                [
                    row.get::<_, f64>(9)? as f32,
                    row.get::<_, f64>(10)? as f32,
                    row.get::<_, f64>(11)? as f32,
                ],
                [0.0, 0.0, 1.0],
            ),
            color: color_bytes([
                row.get::<_, f64>(12)? as f32,
                row.get::<_, f64>(13)? as f32,
                row.get::<_, f64>(14)? as f32,
                row.get::<_, f64>(15)? as f32,
            ]),
            color2: color_bytes([
                row.get::<_, f64>(16)? as f32,
                row.get::<_, f64>(17)? as f32,
                row.get::<_, f64>(18)? as f32,
                row.get::<_, f64>(19)? as f32,
            ]),
            uv1,
            uv2,
            uv3,
            bone_ids,
            bone_weights: normalize_weight_bytes(weights),
            flow: [
                row.get::<_, f64>(42)? as f32,
                row.get::<_, f64>(43)? as f32,
                0.0,
            ],
        })
    })?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(Into::into)
}

fn load_indices(conn: &Connection, mesh_id: i64, part_id: i64) -> Result<Vec<u32>> {
    let mut stmt = conn
        .prepare("SELECT vertex_id FROM indices WHERE mesh = ?1 AND part = ?2 ORDER BY index_id")?;
    let rows = stmt.query_map((mesh_id, part_id), |row| row.get::<_, i64>(0))?;
    let mut indices = Vec::new();
    for row in rows {
        let value = row?;
        if value < 0 {
            bail!("mesh {mesh_id} part {part_id} has negative vertex index {value}");
        }
        indices.push(value as u32);
    }
    Ok(indices)
}

fn load_shape_vertices(
    conn: &Connection,
    mesh_id: i64,
    part_id: i64,
) -> Result<BTreeMap<String, Vec<ShapeVertex>>> {
    let mut stmt = conn.prepare(
        "SELECT shape, vertex_id, position_x, position_y, position_z
         FROM shape_vertices WHERE mesh = ?1 AND part = ?2 ORDER BY shape, vertex_id",
    )?;
    let rows = stmt.query_map((mesh_id, part_id), |row| {
        Ok((
            row.get::<_, String>(0)?,
            ShapeVertex {
                vertex_id: row.get::<_, i64>(1)?.max(0) as usize,
                position: [
                    row.get::<_, f64>(2)? as f32,
                    row.get::<_, f64>(3)? as f32,
                    row.get::<_, f64>(4)? as f32,
                ],
            },
        ))
    })?;
    let mut out: BTreeMap<String, Vec<ShapeVertex>> = BTreeMap::new();
    for row in rows {
        let (shape, vertex) = row?;
        out.entry(shape).or_default().push(vertex);
    }
    Ok(out)
}

fn compile_model_metadata(model: &mut ImportModel) -> Result<()> {
    let mut attributes = Vec::<String>::new();
    let mut attr_ids = HashMap::<String, usize>::new();
    let mut bones = Vec::<String>::new();
    let mut bone_ids = HashMap::<String, usize>::new();
    let mut shape_names = Vec::<String>::new();
    let mut shape_ids = HashMap::<String, usize>::new();
    let mut usage = UsageInfo {
        max_uv: 1,
        ..UsageInfo::default()
    };

    for mesh in &mut model.meshes {
        mesh.bone_global_indices.clear();
        if !mesh.bones.is_empty() {
            usage.has_weights = true;
        }
        for bone in &mesh.bones {
            let id = if let Some(id) = bone_ids.get(bone) {
                *id
            } else {
                let id = bones.len();
                bones.push(bone.clone());
                bone_ids.insert(bone.clone(), id);
                id
            };
            mesh.bone_global_indices.push(id);
        }

        mesh.base_vertex_count = 0;
        mesh.index_count = 0;
        let mut part_vertex_start = 0usize;
        for part in &mut mesh.parts {
            part.vertex_start = part_vertex_start;
            part_vertex_start += part.vertices.len();
            mesh.base_vertex_count += part.vertices.len();
            mesh.index_count += part.indices.len();
            for attribute in &part.attributes {
                if !attr_ids.contains_key(attribute) {
                    if attributes.len() >= 32 {
                        bail!("MDL part attributes are limited to 32 unique names");
                    }
                    attr_ids.insert(attribute.clone(), attributes.len());
                    attributes.push(attribute.clone());
                }
            }
            for (shape, verts) in &part.shape_vertices {
                if !shape_ids.contains_key(shape) {
                    shape_ids.insert(shape.clone(), shape_names.len());
                    shape_names.push(shape.clone());
                }
                if !verts.is_empty() {
                    // shape usage handled below in compile_shape_parts
                }
            }
            for vertex in &part.vertices {
                if vertex.bone_weights[4..].iter().any(|value| *value > 0)
                    || vertex.bone_ids[4..].iter().any(|value| *value > 0)
                {
                    usage.needs_eight_weights = true;
                }
                if vertex.uv3 != [0.0, 1.0] && vertex.uv3 != [0.0, 0.0] {
                    usage.max_uv = usage.max_uv.max(3);
                } else if vertex.uv2 != [0.0, 1.0] && vertex.uv2 != [0.0, 0.0] {
                    usage.max_uv = usage.max_uv.max(2);
                }
                if vertex.color2 != [0, 0, 0, 255] {
                    usage.uses_color2 = true;
                }
                if vertex.flow[0].abs() > f32::EPSILON
                    || vertex.flow[1].abs() > f32::EPSILON
                    || vertex.flow[2].abs() > f32::EPSILON
                {
                    usage.uses_flow = true;
                }
            }
        }
    }

    model.attributes = attributes;
    model.bones = bones;
    model.shape_names = shape_names;
    model.usage = usage;
    compile_shape_parts(model)?;
    Ok(())
}

fn compile_shape_parts(model: &mut ImportModel) -> Result<()> {
    let shape_names = model.shape_names.clone();
    let mut compiled = Vec::new();
    let mut value_count = 0usize;
    for shape in &shape_names {
        for (mesh_index, mesh) in model.meshes.iter_mut().enumerate() {
            let mut replacements = Vec::new();
            let mut next_shape_vertex = mesh.base_vertex_count;
            for part in &mesh.parts {
                let Some(shape_vertices) = part.shape_vertices.get(shape) else {
                    continue;
                };
                let mut replacement_by_base = HashMap::new();
                for shape_vertex in shape_vertices {
                    if shape_vertex.vertex_id >= part.vertices.len() {
                        continue;
                    }
                    let replacement_index = next_shape_vertex;
                    next_shape_vertex += 1;
                    replacement_by_base.insert(
                        part.vertex_start + shape_vertex.vertex_id,
                        replacement_index,
                    );
                }
                if replacement_by_base.is_empty() {
                    continue;
                }
                for index in &part.indices {
                    let base_vertex = part.vertex_start + *index as usize;
                    if let Some(replacement_vertex) = replacement_by_base.get(&base_vertex) {
                        let mesh_index_position = mesh_index_position_for_part(mesh, part, index)?;
                        replacements.push((mesh_index_position, *replacement_vertex));
                    }
                }
            }
            if !replacements.is_empty() {
                value_count += replacements.len();
                compiled.push(CompiledShapePart {
                    shape_name: shape.clone(),
                    mesh_index,
                    replacements,
                });
            }
            mesh.total_vertex_count = next_shape_vertex;
        }
    }
    for mesh in &mut model.meshes {
        if mesh.total_vertex_count == 0 {
            mesh.total_vertex_count = mesh.base_vertex_count;
        }
    }
    model.shape_parts = compiled;
    model.shape_value_count = value_count;
    Ok(())
}

fn mesh_index_position_for_part(
    mesh: &ImportMesh,
    part: &ImportPart,
    target_index: &u32,
) -> Result<usize> {
    let mut offset = 0usize;
    for candidate in &mesh.parts {
        if candidate.part_id == part.part_id {
            for (relative, index) in candidate.indices.iter().enumerate() {
                if index == target_index {
                    return Ok(offset + relative);
                }
            }
            bail!("shape vertex references part vertex that is not used by indices");
        }
        offset += candidate.indices.len();
    }
    bail!("shape part was not found in parent mesh")
}

fn make_uncompressed_mdl(
    model: &ImportModel,
    base: &ModelFile,
    _options: &WrapOptions,
) -> Result<Vec<u8>> {
    let bounds = calculate_bounds(model);
    let layouts = build_vertex_layouts(model, base)?;
    let mut vertex_info_block = Vec::with_capacity(model.meshes.len() * VERTEX_DECLARATION_BYTES);
    for layout in &layouts {
        write_vertex_declaration(&mut vertex_info_block, layout);
    }

    let (geometry, mesh_summaries) = build_geometry(model, &layouts)?;
    if geometry.vertex_data0.len() + geometry.vertex_data1.len() + geometry.vertex_data2.len()
        > MAX_VERTEX_BUFFER_SIZE
    {
        bail!("vertex buffer is too large");
    }
    let vertex_data_block = [
        geometry.vertex_data0,
        geometry.vertex_data1,
        geometry.vertex_data2,
    ]
    .concat();
    let index_data_block = geometry.index_data;

    let (path_info_block, path_offsets) = build_path_info_block(model);
    let bone_sets_block = build_bone_sets_block(model, base.header.version);
    let bone_set_size = if model.usage.has_weights && base.header.version > 0x0100_0005 {
        v6_bone_set_data_size(&bone_sets_block, model.meshes.len())
    } else {
        0
    };
    let shape_data_block = build_shape_data_block(model, &mesh_summaries, &path_offsets)?;
    let part_bone_sets_block = build_part_bone_sets_block(model);
    let bounding_box_data_block = build_bounding_boxes(model, bounds);
    let extra_meshes_block = Vec::<u8>::new();
    let preserved_blocks = PreservedBlocks {
        element_ids: base.data.element_id_data.clone(),
        terrain_shadow_meshes: base.data.terrain_shadow_mesh_data.clone(),
        terrain_shadow_submeshes: base.data.terrain_shadow_submesh_data.clone(),
        neck_morph: remap_neck_morph_data(model, base),
        patch72: Vec::new(),
        padding: build_padding_block(base)?,
    };

    let basic_model_block =
        build_basic_model_block(model, base, bounds, bone_set_size, &preserved_blocks)?;
    let mesh_data_block = build_mesh_data_block(model, &mesh_summaries, &layouts)?;
    let attribute_offset_block = offsets_to_bytes(&path_offsets.attribute_offsets)?;
    let mesh_part_data_block = build_mesh_part_data_block(model, &mesh_summaries)?;
    let material_offset_block = offsets_to_bytes(&path_offsets.material_offsets)?;
    let bone_offset_block = offsets_to_bytes(&path_offsets.bone_offsets)?;

    let combined_data_block_size = MDL_HEADER_SIZE
        + vertex_info_block.len()
        + path_info_block.len()
        + basic_model_block.len()
        + preserved_blocks.element_ids.len()
        + (60 * 3)
        + extra_meshes_block.len()
        + mesh_data_block.len()
        + attribute_offset_block.len()
        + preserved_blocks.terrain_shadow_meshes.len()
        + mesh_part_data_block.len()
        + preserved_blocks.terrain_shadow_submeshes.len()
        + material_offset_block.len()
        + bone_offset_block.len()
        + bone_sets_block.len()
        + shape_data_block.len()
        + part_bone_sets_block.len()
        + preserved_blocks.neck_morph.len()
        + preserved_blocks.patch72.len()
        + preserved_blocks.padding.len()
        + bounding_box_data_block.len();

    let vertex_data_offset = combined_data_block_size as u32;
    let index_data_offset = vertex_data_offset + vertex_data_block.len() as u32;
    let lod_data_block = build_lod_data_block(
        model,
        base,
        vertex_data_offset,
        vertex_data_block.len() as u32,
        index_data_offset,
        index_data_block.len() as u32,
        !preserved_blocks.neck_morph.is_empty(),
    );

    let mut model_data_block = Vec::new();
    model_data_block.extend_from_slice(&path_info_block);
    model_data_block.extend_from_slice(&basic_model_block);
    model_data_block.extend_from_slice(&preserved_blocks.element_ids);
    model_data_block.extend_from_slice(&lod_data_block);
    model_data_block.extend_from_slice(&extra_meshes_block);
    model_data_block.extend_from_slice(&mesh_data_block);
    model_data_block.extend_from_slice(&attribute_offset_block);
    model_data_block.extend_from_slice(&preserved_blocks.terrain_shadow_meshes);
    model_data_block.extend_from_slice(&mesh_part_data_block);
    model_data_block.extend_from_slice(&preserved_blocks.terrain_shadow_submeshes);
    model_data_block.extend_from_slice(&material_offset_block);
    model_data_block.extend_from_slice(&bone_offset_block);
    model_data_block.extend_from_slice(&bone_sets_block);
    model_data_block.extend_from_slice(&shape_data_block);
    model_data_block.extend_from_slice(&part_bone_sets_block);
    model_data_block.extend_from_slice(&preserved_blocks.neck_morph);
    model_data_block.extend_from_slice(&preserved_blocks.patch72);
    model_data_block.extend_from_slice(&preserved_blocks.padding);
    model_data_block.extend_from_slice(&bounding_box_data_block);

    if combined_data_block_size
        != model_data_block.len() + MDL_HEADER_SIZE + vertex_info_block.len()
    {
        bail!("internal MDL offset calculation mismatch");
    }

    let mut header = Vec::with_capacity(MDL_HEADER_SIZE);
    write_u32(&mut header, base.header.version)?;
    write_u32(&mut header, vertex_info_block.len() as u32)?;
    write_u32(&mut header, model_data_block.len() as u32)?;
    write_u16(&mut header, model.meshes.len() as u16)?;
    write_u16(&mut header, model.materials.len() as u16)?;
    write_u32(
        &mut header,
        (MDL_HEADER_SIZE + vertex_info_block.len() + model_data_block.len()) as u32,
    )?;
    write_u32(
        &mut header,
        (MDL_HEADER_SIZE
            + vertex_info_block.len()
            + model_data_block.len()
            + vertex_data_block.len()
            + index_data_block.len()) as u32,
    )?;
    write_u32(
        &mut header,
        (MDL_HEADER_SIZE
            + vertex_info_block.len()
            + model_data_block.len()
            + vertex_data_block.len()
            + index_data_block.len()) as u32,
    )?;
    write_u32(
        &mut header,
        (MDL_HEADER_SIZE
            + vertex_info_block.len()
            + model_data_block.len()
            + vertex_data_block.len()) as u32,
    )?;
    write_u32(
        &mut header,
        (MDL_HEADER_SIZE
            + vertex_info_block.len()
            + model_data_block.len()
            + vertex_data_block.len()
            + index_data_block.len()) as u32,
    )?;
    write_u32(
        &mut header,
        (MDL_HEADER_SIZE
            + vertex_info_block.len()
            + model_data_block.len()
            + vertex_data_block.len()
            + index_data_block.len()) as u32,
    )?;
    write_u32(&mut header, vertex_data_block.len() as u32)?;
    write_u32(&mut header, 0)?;
    write_u32(&mut header, 0)?;
    write_u32(&mut header, index_data_block.len() as u32)?;
    write_u32(&mut header, 0)?;
    write_u32(&mut header, 0)?;
    header.push(1);
    header.push(1);
    header.extend_from_slice(&[0, 0]);
    debug_assert_eq!(header.len(), MDL_HEADER_SIZE);

    let mut mdl = header;
    mdl.extend_from_slice(&vertex_info_block);
    mdl.extend_from_slice(&model_data_block);
    mdl.extend_from_slice(&vertex_data_block);
    mdl.extend_from_slice(&index_data_block);
    Ok(mdl)
}

#[derive(Debug, Default)]
struct PathOffsets {
    attribute_offsets: Vec<u32>,
    bone_offsets: Vec<u32>,
    material_offsets: Vec<u32>,
    shape_offsets: Vec<u32>,
}

fn build_path_info_block(model: &ImportModel) -> (Vec<u8>, PathOffsets) {
    let mut strings = Vec::new();
    let mut offsets = PathOffsets::default();
    for value in &model.attributes {
        offsets.attribute_offsets.push(strings.len() as u32);
        strings.extend_from_slice(value.as_bytes());
        strings.push(0);
    }
    for value in &model.bones {
        offsets.bone_offsets.push(strings.len() as u32);
        strings.extend_from_slice(value.as_bytes());
        strings.push(0);
    }
    for value in &model.materials {
        offsets.material_offsets.push(strings.len() as u32);
        strings.extend_from_slice(value.as_bytes());
        strings.push(0);
    }
    for value in &model.shape_names {
        offsets.shape_offsets.push(strings.len() as u32);
        strings.extend_from_slice(value.as_bytes());
        strings.push(0);
    }
    pad_to(&mut strings, 4);
    let path_count = model.attributes.len()
        + model.bones.len()
        + model.materials.len()
        + model.shape_names.len();
    let mut out = Vec::with_capacity(8 + strings.len());
    out.extend_from_slice(&(path_count as u32).to_le_bytes());
    out.extend_from_slice(&(strings.len() as u32).to_le_bytes());
    out.extend_from_slice(&strings);
    (out, offsets)
}

fn build_basic_model_block(
    model: &ImportModel,
    base: &ModelFile,
    bounds: Bounds,
    bone_set_size: usize,
    preserved_blocks: &PreservedBlocks,
) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let use_parts = model.usage.has_weights || model.meshes.iter().any(|mesh| mesh.parts.len() > 1);
    let submesh_count: usize = if use_parts {
        model.meshes.iter().map(|mesh| mesh.parts.len()).sum()
    } else {
        0
    };
    let bone_table_count = if model.usage.has_weights {
        model.meshes.len()
    } else {
        0
    };
    let mut flags2 = base.data.flags2;
    flags2 &= !EXTRA_LOD_ENABLED;
    flags2 &= !BONELESS_PARTS_ENABLED;

    write_f32(&mut out, bounds.radius)?;
    write_u16(
        &mut out,
        model.meshes.len().try_into().context("too many meshes")?,
    )?;
    write_u16(
        &mut out,
        model
            .attributes
            .len()
            .try_into()
            .context("too many attributes")?,
    )?;
    write_u16(
        &mut out,
        submesh_count.try_into().context("too many mesh parts")?,
    )?;
    write_u16(
        &mut out,
        model
            .materials
            .len()
            .try_into()
            .context("too many materials")?,
    )?;
    write_u16(
        &mut out,
        model.bones.len().try_into().context("too many bones")?,
    )?;
    write_u16(
        &mut out,
        bone_table_count
            .try_into()
            .context("too many bone tables")?,
    )?;
    write_u16(
        &mut out,
        model
            .shape_names
            .len()
            .try_into()
            .context("too many shapes")?,
    )?;
    write_u16(
        &mut out,
        model
            .shape_parts
            .len()
            .try_into()
            .context("too many shape parts")?,
    )?;
    write_u16(
        &mut out,
        model
            .shape_value_count
            .try_into()
            .context("too many shape values")?,
    )?;
    out.push(1);
    out.push(base.data.flags1);
    write_u16(
        &mut out,
        (preserved_blocks.element_ids.len() / 32)
            .try_into()
            .context("too many element-id entries")?,
    )?;
    out.push(
        (preserved_blocks.terrain_shadow_meshes.len() / 20)
            .try_into()
            .context("too many terrain-shadow mesh entries")?,
    );
    out.push(flags2);
    write_f32(&mut out, 0.0)?;
    write_f32(&mut out, 0.0)?;
    write_u16(&mut out, 0)?;
    write_u16(
        &mut out,
        (preserved_blocks.terrain_shadow_submeshes.len() / 12)
            .try_into()
            .context("too many terrain-shadow submesh entries")?,
    )?;

    let mut flags3 = base.data.flags3;
    let material_change_index = model
        .meshes
        .iter()
        .position(|mesh| mesh.mesh_type == MeshType::MaterialChange);
    let crest_change_index = model
        .meshes
        .iter()
        .position(|mesh| mesh.mesh_type == MeshType::CrestChange);
    if material_change_index.is_some() {
        flags3 |= 0x02;
    } else {
        flags3 &= !0x02;
    }
    if crest_change_index.is_some() {
        flags3 |= 0x04;
    } else {
        flags3 &= !0x04;
    }
    out.push(flags3);
    out.push(material_change_index.unwrap_or(0).min(u8::MAX as usize) as u8);
    out.push(crest_change_index.unwrap_or(0).min(u8::MAX as usize) as u8);
    out.push(
        (preserved_blocks.neck_morph.len() / 32)
            .try_into()
            .context("too many neck morph entries")?,
    );
    write_u16(&mut out, bone_set_size.try_into().unwrap_or(0))?;
    write_u16(&mut out, base.data.unknown13)?;
    write_u16(&mut out, 0)?;
    write_u16(&mut out, base.data.unknown15)?;
    write_u16(&mut out, base.data.unknown16)?;
    write_u16(&mut out, base.data.unknown17)?;
    Ok(out)
}

fn build_lod_data_block(
    model: &ImportModel,
    base: &ModelFile,
    vertex_data_offset: u32,
    vertex_data_size: u32,
    index_data_offset: u32,
    index_data_size: u32,
    neck_morph_preserved: bool,
) -> Vec<u8> {
    let lod0 = base.data.lods.first();
    let mut out = Vec::new();
    write_u16(&mut out, mesh_type_offset(model, MeshType::Standard) as u16).unwrap();
    write_u16(&mut out, mesh_type_count(model, MeshType::Standard) as u16).unwrap();
    write_f32(&mut out, 0.0).unwrap();
    write_f32(&mut out, 100.0).unwrap();
    write_u16(&mut out, mesh_type_offset(model, MeshType::Water) as u16).unwrap();
    write_u16(&mut out, mesh_type_count(model, MeshType::Water) as u16).unwrap();
    write_u16(&mut out, mesh_type_offset(model, MeshType::Shadow) as u16).unwrap();
    write_u16(&mut out, mesh_type_count(model, MeshType::Shadow) as u16).unwrap();
    write_u16(
        &mut out,
        mesh_type_offset(model, MeshType::TerrainShadow) as u16,
    )
    .unwrap();
    write_u16(
        &mut out,
        mesh_type_count(model, MeshType::TerrainShadow) as u16,
    )
    .unwrap();
    write_u16(&mut out, mesh_type_offset(model, MeshType::Fog) as u16).unwrap();
    write_u16(&mut out, mesh_type_count(model, MeshType::Fog) as u16).unwrap();
    write_u32(&mut out, 0).unwrap();
    write_u32(&mut out, index_data_offset).unwrap();
    write_u32(&mut out, lod0.map(|lod| lod.unknown6).unwrap_or(0)).unwrap();
    let mut unknown7 = lod0.map(|lod| lod.unknown7).unwrap_or(0);
    if !neck_morph_preserved {
        unknown7 &= 0xffff_0000;
    }
    write_u32(&mut out, unknown7).unwrap();
    write_u32(&mut out, vertex_data_size).unwrap();
    write_u32(&mut out, index_data_size).unwrap();
    write_u32(&mut out, vertex_data_offset).unwrap();
    write_u32(&mut out, index_data_offset).unwrap();
    out.extend_from_slice(&[0; 120]);
    out
}

fn build_mesh_data_block(
    model: &ImportModel,
    summaries: &[MeshGeometrySummary],
    layouts: &[VertexLayout],
) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut part_start = 0u16;
    for (mesh_index, mesh) in model.meshes.iter().enumerate() {
        let summary = &summaries[mesh_index];
        let layout = &layouts[mesh_index];
        write_u16(
            &mut out,
            summary
                .total_vertex_count
                .try_into()
                .context("mesh has too many vertices")?,
        )?;
        write_u16(&mut out, 0)?;
        write_u32(
            &mut out,
            summary
                .index_count
                .try_into()
                .context("mesh has too many indices")?,
        )?;
        write_u16(
            &mut out,
            mesh.material_id
                .try_into()
                .context("material index exceeds u16")?,
        )?;
        write_u16(&mut out, part_start)?;
        write_u16(
            &mut out,
            mesh.parts
                .len()
                .try_into()
                .context("mesh has too many parts")?,
        )?;
        write_u16(
            &mut out,
            if model.usage.has_weights {
                mesh_index as u16
            } else {
                255
            },
        )?;
        write_u32(&mut out, summary.index_offset)?;
        write_u32(&mut out, summary.vertex_offsets[0])?;
        write_u32(&mut out, summary.vertex_offsets[1])?;
        write_u32(&mut out, summary.vertex_offsets[2])?;
        out.push(layout.stream_strides[0]);
        out.push(layout.stream_strides[1]);
        out.push(layout.stream_strides[2]);
        let mut stream_flags = layout.stream_count;
        if layout_uses_eight_weights(layout) {
            stream_flags |= 4;
        }
        out.push(stream_flags);
        part_start = part_start
            .checked_add(mesh.parts.len().try_into().context("too many parts")?)
            .context("too many parts")?;
    }
    Ok(out)
}

fn build_mesh_part_data_block(
    model: &ImportModel,
    summaries: &[MeshGeometrySummary],
) -> Result<Vec<u8>> {
    let mut part_summaries = Vec::<MeshPartSummary>::new();
    let mut current_bone_offset = 0i16;
    for (mesh_index, mesh) in model.meshes.iter().enumerate() {
        let summary = &summaries[mesh_index];
        let mut relative_index_offset = 0u32;
        for part in &mesh.parts {
            let attr_mask = attribute_mask(model, part)?;
            part_summaries.push(MeshPartSummary {
                index_offset: summary.index_offset + relative_index_offset,
                index_count: part.indices.len() as u32,
                attribute_mask: attr_mask,
                bone_start: if model.usage.has_weights {
                    current_bone_offset
                } else {
                    -1
                },
                bone_count: if model.usage.has_weights {
                    mesh.bones.len() as u16
                } else {
                    0
                },
            });
            relative_index_offset += part.indices.len() as u32;
            if model.usage.has_weights {
                current_bone_offset = current_bone_offset.saturating_add(mesh.bones.len() as i16);
            }
        }
    }

    let mut out = Vec::new();
    for part in part_summaries {
        write_u32(&mut out, part.index_offset)?;
        write_u32(&mut out, part.index_count)?;
        write_u32(&mut out, part.attribute_mask)?;
        write_i16(&mut out, part.bone_start)?;
        write_u16(&mut out, part.bone_count)?;
    }
    Ok(out)
}

fn build_bone_sets_block(model: &ImportModel, mdl_version: u32) -> Vec<u8> {
    if !model.usage.has_weights {
        return Vec::new();
    }
    let mut out = Vec::new();
    if mdl_version > 0x0100_0005 {
        let sets: Vec<Vec<u8>> = model
            .meshes
            .iter()
            .map(|mesh| {
                let mut data = Vec::with_capacity(mesh.bone_global_indices.len() * 2);
                for bone_index in &mesh.bone_global_indices {
                    data.extend_from_slice(&(*bone_index as u16).to_le_bytes());
                }
                data
            })
            .collect();
        for set in &sets {
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(&((set.len() / 2) as u16).to_le_bytes());
        }
        for (mesh_index, set) in sets.iter().enumerate() {
            let header_location = mesh_index * 4;
            let distance = ((out.len() - header_location) / 4) as u16;
            out.extend_from_slice(set);
            if set.len() % 4 != 0 {
                out.extend_from_slice(&[0, 0]);
            }
            let bytes = distance.to_le_bytes();
            out[header_location] = bytes[0];
            out[header_location + 1] = bytes[1];
        }
    } else {
        for mesh in &model.meshes {
            for index in 0..64 {
                let bone = mesh.bone_global_indices.get(index).copied().unwrap_or(0) as u16;
                out.extend_from_slice(&bone.to_le_bytes());
            }
            out.push(mesh.bone_global_indices.len().min(64) as u8);
            out.extend_from_slice(&[0, 0, 0]);
        }
    }
    out
}

fn v6_bone_set_data_size(block: &[u8], mesh_count: usize) -> usize {
    block.len().saturating_sub(mesh_count * 4) / 2
}

fn build_shape_data_block(
    model: &ImportModel,
    summaries: &[MeshGeometrySummary],
    path_offsets: &PathOffsets,
) -> Result<Vec<u8>> {
    if model.shape_names.is_empty() || model.shape_parts.is_empty() {
        return Ok(Vec::new());
    }
    let mut shape_counts_by_name = HashMap::<&str, usize>::new();
    for part in &model.shape_parts {
        *shape_counts_by_name
            .entry(part.shape_name.as_str())
            .or_default() += 1;
    }
    let mut out = Vec::new();
    let mut running = 0u16;
    for (shape_index, shape_name) in model.shape_names.iter().enumerate() {
        write_u32(&mut out, path_offsets.shape_offsets[shape_index])?;
        write_u16(&mut out, running)?;
        write_u16(&mut out, 0)?;
        write_u16(&mut out, 0)?;
        let count = *shape_counts_by_name.get(shape_name.as_str()).unwrap_or(&0) as u16;
        running = running.saturating_add(count);
        write_u16(&mut out, count)?;
        write_u16(&mut out, 0)?;
        write_u16(&mut out, 0)?;
    }

    let mut value_offset = 0u32;
    for shape_part in &model.shape_parts {
        write_u32(&mut out, summaries[shape_part.mesh_index].index_offset)?;
        write_u32(&mut out, shape_part.replacements.len() as u32)?;
        write_u32(&mut out, value_offset)?;
        value_offset += shape_part.replacements.len() as u32;
    }
    for shape_part in &model.shape_parts {
        for (base_index, replacement_vertex) in &shape_part.replacements {
            write_u16(
                &mut out,
                (*base_index)
                    .try_into()
                    .context("shape base index exceeds u16")?,
            )?;
            write_u16(
                &mut out,
                (*replacement_vertex)
                    .try_into()
                    .context("shape replacement vertex exceeds u16")?,
            )?;
        }
    }
    Ok(out)
}

fn build_part_bone_sets_block(model: &ImportModel) -> Vec<u8> {
    let mut data = Vec::new();
    if model.usage.has_weights {
        for mesh in &model.meshes {
            for local_bone_index in 0..mesh.bones.len() {
                data.extend_from_slice(&(local_bone_index as u16).to_le_bytes());
            }
        }
    }
    let mut out = Vec::with_capacity(4 + data.len());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&data);
    out
}

fn remap_neck_morph_data(model: &ImportModel, base: &ModelFile) -> Vec<u8> {
    if base.data.neck_morph_data.is_empty() {
        return Vec::new();
    }
    let Some(base_bone_table) = base.data.bone_tables.first() else {
        return Vec::new();
    };
    let Some(first_mesh) = model.meshes.first() else {
        return Vec::new();
    };
    if first_mesh.bones.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(base.data.neck_morph_data.len());
    for chunk in base.data.neck_morph_data.chunks_exact(32) {
        out.extend_from_slice(&chunk[..28]);
        let mut remapped = Vec::<u8>::with_capacity(4);
        for (slot, base_boneset_index) in chunk[28..32].iter().copied().enumerate() {
            if slot >= 2 && base_boneset_index == 0 {
                break;
            }
            let Some(global_bone_index) = base_bone_table.get(base_boneset_index as usize) else {
                return Vec::new();
            };
            let base_bone_name = base.data.bone_name(*global_bone_index);
            let Some(new_local_index) = first_mesh
                .bones
                .iter()
                .position(|candidate| candidate == &base_bone_name)
            else {
                return Vec::new();
            };
            let Ok(new_local_index) = u8::try_from(new_local_index) else {
                return Vec::new();
            };
            remapped.push(new_local_index);
        }
        for slot in 0..4 {
            out.push(remapped.get(slot).copied().unwrap_or(0));
        }
    }
    out
}

fn build_padding_block(base: &ModelFile) -> Result<Vec<u8>> {
    let padding_len = base.data.padding_bytes.len();
    let mut out = Vec::with_capacity(1 + padding_len);
    out.push(
        padding_len
            .try_into()
            .context("base MDL padding block is too large")?,
    );
    out.extend_from_slice(&base.data.padding_bytes);
    Ok(out)
}

fn offsets_to_bytes(offsets: &[u32]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(offsets.len() * 4);
    for offset in offsets {
        write_u32(&mut out, *offset)?;
    }
    Ok(out)
}

fn build_geometry(
    model: &ImportModel,
    layouts: &[VertexLayout],
) -> Result<(GeometryData, Vec<MeshGeometrySummary>)> {
    let mut geometry = GeometryData::default();
    let mut summaries = Vec::with_capacity(model.meshes.len());
    for (mesh_index, mesh) in model.meshes.iter().enumerate() {
        let layout = layouts
            .get(mesh_index)
            .with_context(|| format!("missing vertex layout for mesh {mesh_index}"))?;
        let mut mesh_geometry = GeometryData::default();
        let index_offset = (geometry.index_data.len() / 2) as u32;

        for part in &mesh.parts {
            for vertex in &part.vertices {
                write_vertex(&mut mesh_geometry, layout, vertex)?;
            }
        }
        for part in &mesh.parts {
            for shape_vertices in part.shape_vertices.values() {
                for shape_vertex in shape_vertices {
                    let Some(base_vertex) = part.vertices.get(shape_vertex.vertex_id).copied()
                    else {
                        continue;
                    };
                    let mut vertex = base_vertex;
                    vertex.position = shape_vertex.position;
                    write_vertex(&mut mesh_geometry, layout, &vertex)?;
                }
            }
        }

        let vertex_offset0 = geometry.vertex_data0.len() as u32;
        geometry
            .vertex_data0
            .extend_from_slice(&mesh_geometry.vertex_data0);
        let vertex_offset1 = if mesh_geometry.vertex_data1.is_empty() {
            0
        } else {
            geometry.vertex_data0.len() as u32
        };
        geometry
            .vertex_data0
            .extend_from_slice(&mesh_geometry.vertex_data1);
        let vertex_offset2 = if mesh_geometry.vertex_data2.is_empty() {
            0
        } else {
            geometry.vertex_data0.len() as u32
        };
        geometry
            .vertex_data0
            .extend_from_slice(&mesh_geometry.vertex_data2);

        for part in &mesh.parts {
            for index in &part.indices {
                let value = part.vertex_start + *index as usize;
                if value > u16::MAX as usize {
                    bail!("mesh has an index above u16::MAX");
                }
                geometry
                    .index_data
                    .extend_from_slice(&(value as u16).to_le_bytes());
            }
        }
        let index_count = mesh.index_count;
        pad_to(&mut geometry.index_data, 16);
        summaries.push(MeshGeometrySummary {
            vertex_offsets: [vertex_offset0, vertex_offset1, vertex_offset2],
            index_offset,
            total_vertex_count: mesh.total_vertex_count,
            index_count,
        });
    }
    Ok((geometry, summaries))
}

fn write_vertex(
    geometry: &mut GeometryData,
    layout: &VertexLayout,
    vertex: &ImportVertex,
) -> Result<()> {
    let mut stream_data = [
        vec![0u8; layout.stream_strides[0] as usize],
        vec![0u8; layout.stream_strides[1] as usize],
        vec![0u8; layout.stream_strides[2] as usize],
    ];
    for element in &layout.elements {
        let stream_index = element.stream as usize;
        if stream_index >= stream_data.len() {
            bail!("unsupported vertex stream {}", element.stream);
        }
        let mut encoded = Vec::with_capacity(element.data_type.size() as usize);
        write_vertex_element(&mut encoded, *element, vertex)?;
        let offset = element.offset as usize;
        let end = offset + encoded.len();
        if end > stream_data[stream_index].len() {
            bail!(
                "vertex element {:?}/{} exceeds stream {} stride {}",
                element.usage,
                element.usage_index,
                element.stream,
                layout.stream_strides[stream_index]
            );
        }
        stream_data[stream_index][offset..end].copy_from_slice(&encoded);
    }
    geometry.vertex_data0.extend_from_slice(&stream_data[0]);
    geometry.vertex_data1.extend_from_slice(&stream_data[1]);
    geometry.vertex_data2.extend_from_slice(&stream_data[2]);
    Ok(())
}

fn write_vertex_element(
    out: &mut Vec<u8>,
    element: VertexElement,
    vertex: &ImportVertex,
) -> Result<()> {
    match element.usage {
        VertexUsage::Position => write_vec3(out, element.data_type, vertex.position, 1.0),
        VertexUsage::Normal => write_vec3(out, element.data_type, vertex.normal, 0.0),
        VertexUsage::Binormal => write_direction(
            out,
            element.data_type,
            vertex.binormal,
            infer_binormal_handedness(vertex),
        ),
        VertexUsage::Flow => write_direction(out, element.data_type, vertex.flow, false),
        VertexUsage::Color => {
            if !matches!(
                element.data_type,
                VertexDataType::Ubyte4 | VertexDataType::Ubyte4n
            ) {
                bail!("unsupported color vertex data type {:?}", element.data_type);
            }
            if element.usage_index == 0 {
                out.extend_from_slice(&vertex.color);
            } else {
                out.extend_from_slice(&vertex.color2);
            }
            Ok(())
        }
        VertexUsage::TextureCoordinate => {
            if element.usage_index == 0 {
                match element.data_type {
                    VertexDataType::Float2 => write_float2(out, vertex.uv1),
                    VertexDataType::Float4 => {
                        write_float2(out, vertex.uv1)?;
                        write_float2(out, vertex.uv2)
                    }
                    VertexDataType::Half2 => write_half2(out, vertex.uv1),
                    VertexDataType::Half4 => {
                        write_half2(out, vertex.uv1)?;
                        write_half2(out, vertex.uv2)
                    }
                    VertexDataType::Ubyte4n => write_uv_ubyte4n(out, vertex.uv1, vertex.uv2),
                    _ => bail!("unsupported UV vertex data type {:?}", element.data_type),
                }
            } else {
                match element.data_type {
                    VertexDataType::Float2 => write_float2(out, vertex.uv3),
                    VertexDataType::Half2 => write_half2(out, vertex.uv3),
                    _ => bail!(
                        "unsupported secondary UV vertex data type {:?}",
                        element.data_type
                    ),
                }
            }
        }
        VertexUsage::BoneWeight => {
            match element.data_type {
                VertexDataType::Ubyte8 => {
                    out.extend_from_slice(&[
                        vertex.bone_weights[0],
                        vertex.bone_weights[4],
                        vertex.bone_weights[1],
                        vertex.bone_weights[5],
                        vertex.bone_weights[2],
                        vertex.bone_weights[6],
                        vertex.bone_weights[3],
                        vertex.bone_weights[7],
                    ]);
                }
                VertexDataType::Ubyte4 | VertexDataType::Ubyte4n => {
                    out.extend_from_slice(&vertex.bone_weights[0..4]);
                }
                _ => bail!(
                    "unsupported bone-weight vertex data type {:?}",
                    element.data_type
                ),
            }
            Ok(())
        }
        VertexUsage::BoneIndex => {
            match element.data_type {
                VertexDataType::Ubyte8 => {
                    out.extend_from_slice(&[
                        vertex.bone_ids[0],
                        vertex.bone_ids[4],
                        vertex.bone_ids[1],
                        vertex.bone_ids[5],
                        vertex.bone_ids[2],
                        vertex.bone_ids[6],
                        vertex.bone_ids[3],
                        vertex.bone_ids[7],
                    ]);
                }
                VertexDataType::Ubyte4 | VertexDataType::Ubyte4n => {
                    out.extend_from_slice(&vertex.bone_ids[0..4]);
                }
                _ => bail!(
                    "unsupported bone-index vertex data type {:?}",
                    element.data_type
                ),
            }
            Ok(())
        }
    }
}

fn build_vertex_layouts(model: &ImportModel, base: &ModelFile) -> Result<Vec<VertexLayout>> {
    let mut layouts = Vec::with_capacity(model.meshes.len());
    for (mesh_index, mesh) in model.meshes.iter().enumerate() {
        let mesh_usage = usage_for_mesh(mesh);
        let base_layout = base
            .lod0_meshes
            .get(mesh_index)
            .and_then(|base_mesh| {
                base.data
                    .vertex_declarations
                    .get(base_mesh.original_mesh_index)
            })
            .and_then(|declaration| {
                vertex_layout_from_mdl(declaration)
                    .filter(|layout| layout_supports_usage(layout, mesh_usage))
            });
        layouts.push(base_layout.unwrap_or_else(|| build_vertex_layout(mesh_usage)));
    }
    Ok(layouts)
}

fn usage_for_mesh(mesh: &ImportMesh) -> UsageInfo {
    let mut usage = UsageInfo {
        has_weights: !mesh.bones.is_empty(),
        max_uv: 1,
        ..UsageInfo::default()
    };
    for part in &mesh.parts {
        for vertex in &part.vertices {
            if vertex.bone_weights[4..].iter().any(|value| *value > 0)
                || vertex.bone_ids[4..].iter().any(|value| *value > 0)
            {
                usage.needs_eight_weights = true;
            }
            if vertex.uv3 != [0.0, 1.0] && vertex.uv3 != [0.0, 0.0] {
                usage.max_uv = usage.max_uv.max(3);
            } else if vertex.uv2 != [0.0, 1.0] && vertex.uv2 != [0.0, 0.0] {
                usage.max_uv = usage.max_uv.max(2);
            }
            if vertex.color2 != [0, 0, 0, 255] {
                usage.uses_color2 = true;
            }
            if vertex.flow[0].abs() > f32::EPSILON
                || vertex.flow[1].abs() > f32::EPSILON
                || vertex.flow[2].abs() > f32::EPSILON
            {
                usage.uses_flow = true;
            }
        }
    }
    usage
}

fn vertex_layout_from_mdl(declaration: &MdlVertexDeclaration) -> Option<VertexLayout> {
    let mut stream_strides = [0u8; 3];
    let mut stream_count = 0u8;
    let mut elements = Vec::with_capacity(declaration.elements.len());
    for element in &declaration.elements {
        let stream = element.stream as usize;
        if stream >= stream_strides.len() {
            return None;
        }
        let data_type = VertexDataType::from_mdl(element.vertex_type);
        let end = element.offset.checked_add(data_type.size())?;
        stream_strides[stream] = stream_strides[stream].max(end);
        stream_count = stream_count.max(element.stream + 1);
        elements.push(VertexElement {
            stream: element.stream,
            offset: element.offset,
            data_type,
            usage: VertexUsage::from_mdl(element.usage),
            usage_index: element.usage_index,
        });
    }
    Some(VertexLayout {
        elements,
        stream_strides,
        stream_count,
    })
}

fn layout_supports_usage(layout: &VertexLayout, usage: UsageInfo) -> bool {
    layout.elements.iter().all(element_writable)
        && has_layout_usage(layout, VertexUsage::Position, 0)
        && has_layout_usage(layout, VertexUsage::Normal, 0)
        && has_layout_usage(layout, VertexUsage::Binormal, 0)
        && has_layout_usage(layout, VertexUsage::TextureCoordinate, 0)
        && (!usage.has_weights
            || (has_layout_usage(layout, VertexUsage::BoneWeight, 0)
                && has_layout_usage(layout, VertexUsage::BoneIndex, 0)))
        && (!usage.needs_eight_weights || layout_uses_eight_weights(layout))
        && (usage.max_uv <= 2 || has_layout_usage(layout, VertexUsage::TextureCoordinate, 1))
        && (!usage.uses_color2 || has_layout_usage(layout, VertexUsage::Color, 1))
        && (!usage.uses_flow || has_layout_usage(layout, VertexUsage::Flow, 0))
}

fn element_writable(element: &VertexElement) -> bool {
    match element.usage {
        VertexUsage::Position | VertexUsage::Normal => matches!(
            element.data_type,
            VertexDataType::Float3
                | VertexDataType::Float4
                | VertexDataType::Half4
                | VertexDataType::Ubyte4n
        ),
        VertexUsage::Binormal | VertexUsage::Flow => matches!(
            element.data_type,
            VertexDataType::Ubyte4
                | VertexDataType::Ubyte4n
                | VertexDataType::Float3
                | VertexDataType::Float4
                | VertexDataType::Half4
        ),
        VertexUsage::Color => matches!(
            element.data_type,
            VertexDataType::Ubyte4 | VertexDataType::Ubyte4n
        ),
        VertexUsage::TextureCoordinate if element.usage_index == 0 => matches!(
            element.data_type,
            VertexDataType::Float2
                | VertexDataType::Float4
                | VertexDataType::Half2
                | VertexDataType::Half4
                | VertexDataType::Ubyte4n
        ),
        VertexUsage::TextureCoordinate => {
            matches!(
                element.data_type,
                VertexDataType::Float2 | VertexDataType::Half2
            )
        }
        VertexUsage::BoneWeight | VertexUsage::BoneIndex => matches!(
            element.data_type,
            VertexDataType::Ubyte4 | VertexDataType::Ubyte4n | VertexDataType::Ubyte8
        ),
    }
}

fn has_layout_usage(layout: &VertexLayout, usage: VertexUsage, usage_index: u8) -> bool {
    layout
        .elements
        .iter()
        .any(|element| element.usage == usage && element.usage_index == usage_index)
}

fn layout_uses_eight_weights(layout: &VertexLayout) -> bool {
    layout.elements.iter().any(|element| {
        matches!(
            (element.usage, element.data_type),
            (
                VertexUsage::BoneWeight | VertexUsage::BoneIndex,
                VertexDataType::Ubyte8
            )
        )
    })
}

fn build_vertex_layout(usage: UsageInfo) -> VertexLayout {
    let mut elements = Vec::new();
    let mut offsets = [0u8; 3];
    push_element(
        &mut elements,
        &mut offsets,
        0,
        VertexDataType::Float3,
        VertexUsage::Position,
        0,
    );
    if usage.has_weights {
        push_element(
            &mut elements,
            &mut offsets,
            0,
            if usage.needs_eight_weights {
                VertexDataType::Ubyte8
            } else {
                VertexDataType::Ubyte4n
            },
            VertexUsage::BoneWeight,
            0,
        );
        push_element(
            &mut elements,
            &mut offsets,
            0,
            if usage.needs_eight_weights {
                VertexDataType::Ubyte8
            } else {
                VertexDataType::Ubyte4
            },
            VertexUsage::BoneIndex,
            0,
        );
    }
    push_element(
        &mut elements,
        &mut offsets,
        1,
        VertexDataType::Float3,
        VertexUsage::Normal,
        0,
    );
    push_element(
        &mut elements,
        &mut offsets,
        1,
        VertexDataType::Ubyte4n,
        VertexUsage::Binormal,
        0,
    );
    if usage.uses_flow {
        push_element(
            &mut elements,
            &mut offsets,
            1,
            VertexDataType::Ubyte4n,
            VertexUsage::Flow,
            0,
        );
    }
    push_element(
        &mut elements,
        &mut offsets,
        1,
        VertexDataType::Ubyte4n,
        VertexUsage::Color,
        0,
    );
    if usage.uses_color2 {
        push_element(
            &mut elements,
            &mut offsets,
            1,
            VertexDataType::Ubyte4n,
            VertexUsage::Color,
            1,
        );
    }
    push_element(
        &mut elements,
        &mut offsets,
        1,
        if usage.max_uv > 1 {
            VertexDataType::Float4
        } else {
            VertexDataType::Float2
        },
        VertexUsage::TextureCoordinate,
        0,
    );
    if usage.max_uv > 2 {
        push_element(
            &mut elements,
            &mut offsets,
            1,
            VertexDataType::Float2,
            VertexUsage::TextureCoordinate,
            1,
        );
    }
    VertexLayout {
        elements,
        stream_strides: offsets,
        stream_count: 2,
    }
}

fn push_element(
    elements: &mut Vec<VertexElement>,
    offsets: &mut [u8; 3],
    stream: u8,
    data_type: VertexDataType,
    usage: VertexUsage,
    usage_index: u8,
) {
    let offset = offsets[stream as usize];
    elements.push(VertexElement {
        stream,
        offset,
        data_type,
        usage,
        usage_index,
    });
    offsets[stream as usize] += data_type.size();
}

fn write_vertex_declaration(out: &mut Vec<u8>, layout: &VertexLayout) {
    let start = out.len();
    for element in &layout.elements {
        out.push(element.stream);
        out.push(element.offset);
        out.push(element.data_type as u8);
        out.push(element.usage as u8);
        out.push(element.usage_index);
        out.extend_from_slice(&[0, 0, 0]);
    }
    out.push(0xFF);
    let written = out.len() - start;
    if written < VERTEX_DECLARATION_BYTES {
        out.extend(std::iter::repeat_n(0, VERTEX_DECLARATION_BYTES - written));
    }
}

fn attribute_mask(model: &ImportModel, part: &ImportPart) -> Result<u32> {
    let mut mask = 0u32;
    for attribute in &part.attributes {
        let Some(index) = model
            .attributes
            .iter()
            .position(|candidate| candidate == attribute)
        else {
            continue;
        };
        if index >= 32 {
            bail!("attribute index exceeds mask width");
        }
        mask |= 1u32 << index;
    }
    Ok(mask)
}

fn calculate_bounds(model: &ImportModel) -> Bounds {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let mut any = false;
    for mesh in &model.meshes {
        for part in &mesh.parts {
            for vertex in &part.vertices {
                any = true;
                for axis in 0..3 {
                    min[axis] = min[axis].min(vertex.position[axis]);
                    max[axis] = max[axis].max(vertex.position[axis]);
                }
            }
        }
    }
    if !any {
        return Bounds {
            min: [0.0; 3],
            max: [0.0; 3],
            radius: 0.0,
        };
    }
    let radius = max
        .iter()
        .zip(min.iter())
        .flat_map(|(max, min)| [max.abs(), min.abs()])
        .fold(0.0f32, f32::max)
        * 3.0f32.sqrt();
    Bounds { min, max, radius }
}

fn build_bounding_boxes(model: &ImportModel, bounds: Bounds) -> Vec<u8> {
    let mut out = Vec::new();
    for index in 0..4 {
        if index < 2 {
            let min = if index == 0 {
                [
                    bounds.min[0].min(0.0),
                    bounds.min[1].min(0.0),
                    bounds.min[2].min(0.0),
                ]
            } else {
                bounds.min
            };
            let max = if index == 0 {
                [
                    bounds.max[0].max(0.0),
                    bounds.max[1].max(0.0),
                    bounds.max[2].max(0.0),
                ]
            } else {
                bounds.max
            };
            for value in min {
                out.extend_from_slice(&value.to_le_bytes());
            }
            out.extend_from_slice(&1.0f32.to_le_bytes());
            for value in max {
                out.extend_from_slice(&value.to_le_bytes());
            }
            out.extend_from_slice(&1.0f32.to_le_bytes());
        } else {
            out.extend_from_slice(&[0; 32]);
        }
    }
    let divisor = 20.0f32;
    for _ in &model.bones {
        for value in [
            -bounds.radius / divisor,
            -bounds.radius / divisor,
            -bounds.radius / divisor,
        ] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.extend_from_slice(&1.0f32.to_le_bytes());
        for value in [
            bounds.radius / divisor,
            bounds.radius / divisor,
            bounds.radius / divisor,
        ] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.extend_from_slice(&1.0f32.to_le_bytes());
    }
    out
}

fn mesh_type_offset(model: &ImportModel, mesh_type: MeshType) -> usize {
    model
        .meshes
        .iter()
        .position(|mesh| mesh.mesh_type == mesh_type)
        .unwrap_or(0)
}

fn mesh_type_count(model: &ImportModel, mesh_type: MeshType) -> usize {
    model
        .meshes
        .iter()
        .filter(|mesh| mesh.mesh_type == mesh_type)
        .count()
}

fn normalize_material_path(name: &str) -> String {
    let mut value = name.trim().replace('\\', "/");
    if !value.starts_with('/') {
        value.insert(0, '/');
    }
    if !value.to_ascii_lowercase().ends_with(".mtrl") {
        value.push_str(".mtrl");
    }
    value
}

fn split_attributes(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn import_uv(mut uv: [f32; 2], shift_import_uv: bool) -> [f32; 2] {
    if shift_import_uv {
        uv[1] = -uv[1] + 1.0;
    }
    uv
}

fn color_bytes(values: [f32; 4]) -> [u8; 4] {
    values.map(float_to_byte)
}

fn float_to_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn normalize_weight_bytes(weights: [f32; 8]) -> [u8; 8] {
    let mut raw = [0u8; 8];
    let clamped = weights.map(|value| value.max(0.0));
    let sum: f32 = clamped.iter().sum();
    if sum <= f32::EPSILON {
        raw[0] = 255;
        return raw;
    }
    let mut running = 0i32;
    for index in 0..7 {
        raw[index] = ((clamped[index] / sum) * 255.0).round().clamp(0.0, 255.0) as u8;
        running += raw[index] as i32;
    }
    raw[7] = (255 - running).clamp(0, 255) as u8;
    raw
}

fn write_vec3(out: &mut Vec<u8>, data_type: VertexDataType, value: [f32; 3], w: f32) -> Result<()> {
    match data_type {
        VertexDataType::Float3 => {
            for item in value {
                write_f32(out, item)?;
            }
        }
        VertexDataType::Float4 => {
            for item in value {
                write_f32(out, item)?;
            }
            write_f32(out, w)?;
        }
        VertexDataType::Half4 => {
            for item in value {
                out.extend_from_slice(&f16::from_f32(item).to_bits().to_le_bytes());
            }
            out.extend_from_slice(&f16::from_f32(w).to_bits().to_le_bytes());
        }
        VertexDataType::Ubyte4n => write_packed_vec3(out, value, w < 0.0),
        _ => bail!("unsupported vec3 vertex data type {:?}", data_type),
    }
    Ok(())
}

fn write_float2(out: &mut Vec<u8>, value: [f32; 2]) -> Result<()> {
    write_f32(out, value[0])?;
    write_f32(out, value[1])?;
    Ok(())
}

fn write_half2(out: &mut Vec<u8>, value: [f32; 2]) -> Result<()> {
    out.extend_from_slice(&f16::from_f32(value[0]).to_bits().to_le_bytes());
    out.extend_from_slice(&f16::from_f32(value[1]).to_bits().to_le_bytes());
    Ok(())
}

fn write_direction(
    out: &mut Vec<u8>,
    data_type: VertexDataType,
    value: [f32; 3],
    handedness: bool,
) -> Result<()> {
    match data_type {
        VertexDataType::Ubyte4 | VertexDataType::Ubyte4n => {
            write_packed_vec3(out, value, handedness);
            Ok(())
        }
        VertexDataType::Float3 => write_vec3(out, data_type, value, 0.0),
        VertexDataType::Float4 | VertexDataType::Half4 => {
            write_vec3(out, data_type, value, if handedness { -1.0 } else { 1.0 })
        }
        _ => bail!("unsupported direction vertex data type {:?}", data_type),
    }
}

fn write_uv_ubyte4n(out: &mut Vec<u8>, uv1: [f32; 2], uv2: [f32; 2]) -> Result<()> {
    for value in [uv1[0], uv1[1], uv2[0], uv2[1]] {
        out.push(float_to_byte(value));
    }
    Ok(())
}

fn write_packed_vec3(out: &mut Vec<u8>, value: [f32; 3], handedness: bool) {
    for item in value {
        out.push(((item.clamp(-1.0, 1.0) + 1.0) * 127.5).round() as u8);
    }
    out.push(if handedness { 255 } else { 0 });
}

fn infer_binormal_handedness(vertex: &ImportVertex) -> bool {
    dot3(vertex.tangent, cross3(vertex.normal, vertex.binormal)) < 0.0
}

fn normalize3_or(value: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
    if len <= f32::EPSILON {
        fallback
    } else {
        [value[0] / len, value[1] / len, value[2] / len]
    }
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn pad_to(buffer: &mut Vec<u8>, align: usize) {
    let padding = (align - (buffer.len() % align)) % align;
    buffer.extend(std::iter::repeat_n(0, padding));
}

fn write_u16(out: &mut Vec<u8>, value: u16) -> Result<()> {
    out.write_u16::<LittleEndian>(value)?;
    Ok(())
}

fn write_i16(out: &mut Vec<u8>, value: i16) -> Result<()> {
    out.write_i16::<LittleEndian>(value)?;
    Ok(())
}

fn write_u32(out: &mut Vec<u8>, value: u32) -> Result<()> {
    out.write_u32::<LittleEndian>(value)?;
    Ok(())
}

fn write_f32(out: &mut Vec<u8>, value: f32) -> Result<()> {
    out.write_f32::<LittleEndian>(value)?;
    Ok(())
}

fn temp_mdl_path(dest_mdl: &Path) -> std::path::PathBuf {
    let file_name = dest_mdl
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("output.mdl");
    dest_mdl.with_file_name(format!(".{file_name}.tmp"))
}
