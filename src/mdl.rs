// SPDX-License-Identifier: GPL-3.0-or-later
#![allow(dead_code)]

use std::collections::HashMap;
use std::fs;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context, Result, bail};
use byteorder::{LittleEndian, ReadBytesExt};
use half::f16;

const VERTEX_DECLARATION_SLOTS: usize = 17;
const VERTEX_ELEMENT_SIZE: u64 = 8;
const EXTRA_LOD_ENABLED: u8 = 0x10;

#[derive(Debug, Clone)]
pub struct ModelFile {
    pub source_stem: String,
    pub header: ModelFileHeader,
    pub data: ModelData,
    pub lod0_meshes: Vec<DecodedMesh>,
}

#[derive(Debug, Clone)]
pub struct ModelFileHeader {
    pub version: u32,
    pub vertex_declaration_count: u16,
    pub material_count: u16,
    pub vertex_offsets: [u32; 3],
    pub index_offsets: [u32; 3],
    pub vertex_buffer_size: [u32; 3],
    pub index_buffer_size: [u32; 3],
    pub lod_count: u8,
}

#[derive(Debug, Clone)]
pub struct ModelData {
    pub vertex_declarations: Vec<VertexDeclaration>,
    pub strings: Vec<u8>,
    pub mesh_count: u16,
    pub attribute_count: u16,
    pub submesh_count: u16,
    pub material_count: u16,
    pub bone_count: u16,
    pub bone_table_count: u16,
    pub bone_set_size: u16,
    pub shape_count: u16,
    pub shape_mesh_count: u16,
    pub shape_value_count: u16,
    pub lod_count: u8,
    pub flags1: u8,
    pub flags2: u8,
    pub flags3: u8,
    pub element_id_count: u16,
    pub terrain_shadow_mesh_count: u8,
    pub terrain_shadow_submesh_count: u16,
    pub furniture_part_bounding_box_count: u16,
    pub neck_morph_table_size: u8,
    pub patch_72_table_size: u16,
    pub unknown13: u16,
    pub unknown15: u16,
    pub unknown16: u16,
    pub unknown17: u16,
    pub model_clip_out: f32,
    pub shadow_clip_out: f32,
    pub bg_change_material_index: u8,
    pub bg_crest_change_material_index: u8,
    pub lods: Vec<MeshLod>,
    pub meshes: Vec<Mesh>,
    pub attribute_name_offsets: Vec<u32>,
    pub submeshes: Vec<Submesh>,
    pub material_name_offsets: Vec<u32>,
    pub bone_name_offsets: Vec<u32>,
    pub bone_tables: Vec<Vec<u16>>,
    pub shapes: Vec<ShapeStruct>,
    pub shape_meshes: Vec<ShapeMesh>,
    pub shape_values: Vec<ShapeValue>,
    pub element_id_data: Vec<u8>,
    pub extra_lod_data: Vec<u8>,
    pub terrain_shadow_mesh_data: Vec<u8>,
    pub terrain_shadow_submesh_data: Vec<u8>,
    pub submesh_bone_map_data: Vec<u8>,
    pub neck_morph_data: Vec<u8>,
    pub patch_72_data: Vec<u8>,
    pub padding_bytes: Vec<u8>,
    pub model_bounding_box_data: Vec<u8>,
    pub bone_bounding_box_data: Vec<u8>,
    pub furniture_bounding_box_data: Vec<u8>,
}

impl ModelData {
    pub fn string_at(&self, offset: u32) -> String {
        read_c_string(&self.strings, offset as usize)
    }

    pub fn attributes_for_mask(&self, mask: u32) -> Vec<String> {
        self.attribute_name_offsets
            .iter()
            .enumerate()
            .filter_map(|(index, offset)| {
                if mask & (1u32 << index) == 0 {
                    return None;
                }
                let name = self.string_at(*offset);
                if name.is_empty() { None } else { Some(name) }
            })
            .collect()
    }

    pub fn material_name(&self, index: u16) -> String {
        self.material_name_offsets
            .get(index as usize)
            .map(|offset| self.string_at(*offset))
            .unwrap_or_default()
    }

    pub fn bone_name(&self, index: u16) -> String {
        self.bone_name_offsets
            .get(index as usize)
            .map(|offset| self.string_at(*offset))
            .unwrap_or_default()
    }

    pub fn shape_name(&self, shape: &ShapeStruct) -> String {
        self.string_at(shape.string_offset)
    }
}

#[derive(Debug, Clone)]
pub struct VertexDeclaration {
    pub elements: Vec<VertexElement>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexElement {
    pub stream: u8,
    pub offset: u8,
    pub vertex_type: VertexType,
    pub usage: VertexUsage,
    pub usage_index: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum VertexType {
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

impl TryFrom<u8> for VertexType {
    type Error = anyhow::Error;

    fn try_from(value: u8) -> Result<Self> {
        Ok(match value {
            0x00 => Self::Float1,
            0x01 => Self::Float2,
            0x02 => Self::Float3,
            0x03 => Self::Float4,
            0x05 => Self::Ubyte4,
            0x06 => Self::Short2,
            0x07 => Self::Short4,
            0x08 => Self::Ubyte4n,
            0x09 => Self::Short2n,
            0x0A => Self::Short4n,
            0x0B => Self::Ushort2n,
            0x0C => Self::Ushort4n,
            0x0D => Self::Half2,
            0x0E => Self::Half4,
            0x11 => Self::Ubyte8,
            _ => bail!("unsupported vertex type 0x{value:02x}"),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum VertexUsage {
    Position = 0,
    BoneWeight = 1,
    BoneIndex = 2,
    Normal = 3,
    TextureCoordinate = 4,
    Flow = 5,
    Binormal = 6,
    Color = 7,
}

impl TryFrom<u8> for VertexUsage {
    type Error = anyhow::Error;

    fn try_from(value: u8) -> Result<Self> {
        Ok(match value {
            0 => Self::Position,
            1 => Self::BoneWeight,
            2 => Self::BoneIndex,
            3 => Self::Normal,
            4 => Self::TextureCoordinate,
            5 => Self::Flow,
            6 => Self::Binormal,
            7 => Self::Color,
            _ => bail!("unsupported vertex usage 0x{value:02x}"),
        })
    }
}

#[derive(Debug, Clone)]
pub struct MeshLod {
    pub mesh_index: u16,
    pub mesh_count: u16,
    pub model_lod_range: f32,
    pub texture_lod_range: f32,
    pub water_mesh_index: u16,
    pub water_mesh_count: u16,
    pub shadow_mesh_index: u16,
    pub shadow_mesh_count: u16,
    pub terrain_shadow_mesh_index: u16,
    pub terrain_shadow_mesh_count: u16,
    pub vertical_fog_mesh_index: u16,
    pub vertical_fog_mesh_count: u16,
    pub edge_geometry_size: u32,
    pub edge_geometry_data_offset: u32,
    pub vertex_buffer_size: u32,
    pub index_buffer_size: u32,
    pub vertex_data_offset: u32,
    pub index_data_offset: u32,
    pub unknown6: u32,
    pub unknown7: u32,
}

#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertex_count: u16,
    pub index_count: u32,
    pub material_index: u16,
    pub submesh_index: u16,
    pub submesh_count: u16,
    pub bone_table_index: u16,
    pub start_index: u32,
    pub vertex_buffer_offsets: [u32; 3],
    pub vertex_buffer_strides: [u8; 3],
    pub vertex_stream_count: u8,
}

#[derive(Debug, Clone)]
pub struct Submesh {
    pub index_offset: u32,
    pub index_count: u32,
    pub attribute_index_mask: u32,
    pub bone_start_index: u16,
    pub bone_count: u16,
}

#[derive(Debug, Clone)]
pub struct ShapeStruct {
    pub string_offset: u32,
    pub shape_mesh_start_index: [u16; 3],
    pub shape_mesh_count: [u16; 3],
}

#[derive(Debug, Clone)]
pub struct ShapeMesh {
    pub mesh_index_offset: u32,
    pub shape_value_count: u32,
    pub shape_value_offset: u32,
}

#[derive(Debug, Clone)]
pub struct ShapeValue {
    pub base_indices_index: u16,
    pub replacing_vertex_index: u16,
}

#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub binormal: [f32; 3],
    pub binormal_handedness: bool,
    pub tangent: [f32; 3],
    pub color: [f32; 4],
    pub color2: [f32; 4],
    pub uv1: [f32; 2],
    pub uv2: [f32; 2],
    pub uv3: [f32; 2],
    pub bone_ids: [u8; 8],
    pub bone_weights: [f32; 8],
    pub flow: [f32; 3],
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            normal: [0.0; 3],
            binormal: [0.0; 3],
            binormal_handedness: false,
            tangent: [0.0; 3],
            color: [1.0, 1.0, 1.0, 1.0],
            color2: [0.0, 0.0, 0.0, 1.0],
            uv1: [0.0; 2],
            uv2: [0.0; 2],
            uv3: [0.0; 2],
            bone_ids: [0; 8],
            bone_weights: [0.0; 8],
            flow: [0.0; 3],
        }
    }
}

impl Vertex {
    pub fn export_ready(mut self) -> Self {
        self.uv1[1] = -self.uv1[1] + 1.0;
        self.uv2[1] = -self.uv2[1] + 1.0;
        self.uv3[1] = -self.uv3[1] + 1.0;
        self
    }
}

#[derive(Debug, Clone)]
pub struct DecodedMesh {
    pub original_mesh_index: usize,
    pub mesh: Mesh,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u16>,
    pub submeshes: Vec<Submesh>,
    pub bone_names: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct MeshPartExport {
    pub part_index: usize,
    pub submesh: Submesh,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub original_to_part_vertex: HashMap<u16, usize>,
}

impl DecodedMesh {
    pub fn export_parts(&self) -> Result<Vec<MeshPartExport>> {
        let mut parts = Vec::with_capacity(self.submeshes.len());
        for (part_index, submesh) in self.submeshes.iter().cloned().enumerate() {
            let start = submesh
                .index_offset
                .checked_sub(self.mesh.start_index)
                .with_context(|| {
                    format!(
                        "submesh index offset {} precedes mesh start index {}",
                        submesh.index_offset, self.mesh.start_index
                    )
                })? as usize;
            let end = start + submesh.index_count as usize;
            if end > self.indices.len() {
                bail!(
                    "submesh index range {start}..{end} exceeds mesh index count {}",
                    self.indices.len()
                );
            }

            let mut unique_original_vertices = self.indices[start..end].to_vec();
            unique_original_vertices.sort_unstable();
            unique_original_vertices.dedup();

            let mut original_to_part_vertex =
                HashMap::with_capacity(unique_original_vertices.len());
            let mut vertices = Vec::with_capacity(unique_original_vertices.len());
            for (local_id, original_vertex_id) in unique_original_vertices.iter().enumerate() {
                original_to_part_vertex.insert(*original_vertex_id, local_id);
                vertices.push(
                    self.vertices
                        .get(*original_vertex_id as usize)
                        .copied()
                        .unwrap_or_default()
                        .export_ready(),
                );
            }

            let mut indices = Vec::with_capacity(submesh.index_count as usize);
            for original_vertex_id in &self.indices[start..end] {
                let local_id = original_to_part_vertex[original_vertex_id];
                indices.push(local_id as u32);
            }
            calculate_tangents_from_binormals(&mut vertices);
            parts.push(MeshPartExport {
                part_index,
                submesh,
                vertices,
                indices,
                original_to_part_vertex,
            });
        }
        Ok(parts)
    }
}

pub fn parse_mdl(path: &Path) -> Result<ModelFile> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut cursor = Cursor::new(bytes.as_slice());
    let header = read_file_header(&mut cursor)?;
    let data = read_model_data(&mut cursor, &header)?;
    let lod0_meshes = decode_lod_meshes(bytes.as_slice(), &data, 0)?;
    let source_stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("model")
        .to_string();
    Ok(ModelFile {
        source_stem,
        header,
        data,
        lod0_meshes,
    })
}

fn read_file_header(cursor: &mut Cursor<&[u8]>) -> Result<ModelFileHeader> {
    let version = cursor.read_u32::<LittleEndian>()?;
    let _stack_size = cursor.read_u32::<LittleEndian>()?;
    let _runtime_size = cursor.read_u32::<LittleEndian>()?;
    let vertex_declaration_count = cursor.read_u16::<LittleEndian>()?;
    let material_count = cursor.read_u16::<LittleEndian>()?;
    let vertex_offsets = read_u32_array_3(cursor)?;
    let index_offsets = read_u32_array_3(cursor)?;
    let vertex_buffer_size = read_u32_array_3(cursor)?;
    let index_buffer_size = read_u32_array_3(cursor)?;
    let lod_count = cursor.read_u8()?;
    let _index_buffer_streaming_enabled = cursor.read_u8()?;
    let _has_edge_geometry = cursor.read_u8()?;
    let _padding = cursor.read_u8()?;
    Ok(ModelFileHeader {
        version,
        vertex_declaration_count,
        material_count,
        vertex_offsets,
        index_offsets,
        vertex_buffer_size,
        index_buffer_size,
        lod_count,
    })
}

fn read_model_data(cursor: &mut Cursor<&[u8]>, header: &ModelFileHeader) -> Result<ModelData> {
    let vertex_declarations = read_vertex_declarations(cursor, header.vertex_declaration_count)?;
    let _string_count = cursor.read_u16::<LittleEndian>()?;
    let _unknown_string_padding = cursor.read_u16::<LittleEndian>()?;
    let string_size = cursor.read_u32::<LittleEndian>()?;
    let mut strings = vec![0; string_size as usize];
    cursor.read_exact(&mut strings)?;

    let _radius = cursor.read_f32::<LittleEndian>()?;
    let mesh_count = cursor.read_u16::<LittleEndian>()?;
    let attribute_count = cursor.read_u16::<LittleEndian>()?;
    let submesh_count = cursor.read_u16::<LittleEndian>()?;
    let material_count = cursor.read_u16::<LittleEndian>()?;
    let bone_count = cursor.read_u16::<LittleEndian>()?;
    let bone_table_count = cursor.read_u16::<LittleEndian>()?;
    let shape_count = cursor.read_u16::<LittleEndian>()?;
    let shape_mesh_count = cursor.read_u16::<LittleEndian>()?;
    let shape_value_count = cursor.read_u16::<LittleEndian>()?;
    let lod_count = cursor.read_u8()?;
    let flags1 = cursor.read_u8()?;
    let element_id_count = cursor.read_u16::<LittleEndian>()?;
    let terrain_shadow_mesh_count = cursor.read_u8()?;
    let flags2 = cursor.read_u8()?;
    let model_clip_out = cursor.read_f32::<LittleEndian>()?;
    let shadow_clip_out = cursor.read_f32::<LittleEndian>()?;
    let furniture_part_bounding_box_count = cursor.read_u16::<LittleEndian>()?;
    let terrain_shadow_submesh_count = cursor.read_u16::<LittleEndian>()?;
    let flags3 = cursor.read_u8()?;
    let bg_change_material_index = cursor.read_u8()?;
    let bg_crest_change_material_index = cursor.read_u8()?;
    let neck_morph_table_size = cursor.read_u8()?;
    let bone_set_size = cursor.read_u16::<LittleEndian>()?;
    let unknown13 = cursor.read_u16::<LittleEndian>()?;
    let patch_72_table_size = cursor.read_u16::<LittleEndian>()?;
    let unknown15 = cursor.read_u16::<LittleEndian>()?;
    let unknown16 = cursor.read_u16::<LittleEndian>()?;
    let unknown17 = cursor.read_u16::<LittleEndian>()?;

    let element_id_data = read_bytes(cursor, element_id_count as usize * 32)?;

    let mut lods = Vec::with_capacity(3);
    for _ in 0..3 {
        lods.push(read_lod(cursor)?);
    }

    let extra_lod_data = if flags2 & EXTRA_LOD_ENABLED != 0 {
        read_bytes(cursor, 3 * 40)?
    } else {
        Vec::new()
    };

    let mut meshes = Vec::with_capacity(mesh_count as usize);
    for _ in 0..mesh_count {
        meshes.push(read_mesh(cursor)?);
    }

    let attribute_name_offsets = read_u32_vec(cursor, attribute_count as usize)?;
    let terrain_shadow_mesh_data = read_bytes(cursor, terrain_shadow_mesh_count as usize * 20)?;

    let mut submeshes = Vec::with_capacity(submesh_count as usize);
    for _ in 0..submesh_count {
        submeshes.push(read_submesh(cursor)?);
    }

    let terrain_shadow_submesh_data =
        read_bytes(cursor, terrain_shadow_submesh_count as usize * 12)?;
    let material_name_offsets = read_u32_vec(cursor, material_count as usize)?;
    let bone_name_offsets = read_u32_vec(cursor, bone_count as usize)?;
    let bone_tables = read_bone_tables(cursor, header.version, bone_table_count, bone_set_size)?;

    let mut shapes = Vec::with_capacity(shape_count as usize);
    for _ in 0..shape_count {
        shapes.push(read_shape_struct(cursor)?);
    }

    let mut shape_meshes = Vec::with_capacity(shape_mesh_count as usize);
    for _ in 0..shape_mesh_count {
        shape_meshes.push(read_shape_mesh(cursor)?);
    }

    let mut shape_values = Vec::with_capacity(shape_value_count as usize);
    for _ in 0..shape_value_count {
        shape_values.push(read_shape_value(cursor)?);
    }

    let submesh_bone_map_size = cursor.read_u32::<LittleEndian>()?;
    let submesh_bone_map_data = read_bytes(cursor, submesh_bone_map_size as usize)?;
    let neck_morph_data = read_bytes(cursor, neck_morph_table_size as usize * 32)?;
    let patch_72_data = read_bytes(cursor, patch_72_table_size as usize * 16)?;
    let padding_amount = cursor.read_u8()?;
    let padding_bytes = read_bytes(cursor, padding_amount as usize)?;

    let model_bounding_box_data = read_bytes(cursor, 4 * 32)?;
    let bone_bounding_box_data = read_bytes(cursor, bone_count as usize * 32)?;
    let furniture_bounding_box_data =
        read_bytes(cursor, furniture_part_bounding_box_count as usize * 32)?;

    Ok(ModelData {
        vertex_declarations,
        strings,
        mesh_count,
        attribute_count,
        submesh_count,
        material_count,
        bone_count,
        bone_table_count,
        bone_set_size,
        shape_count,
        shape_mesh_count,
        shape_value_count,
        lod_count,
        flags1,
        flags2,
        flags3,
        element_id_count,
        terrain_shadow_mesh_count,
        terrain_shadow_submesh_count,
        furniture_part_bounding_box_count,
        neck_morph_table_size,
        patch_72_table_size,
        unknown13,
        unknown15,
        unknown16,
        unknown17,
        model_clip_out,
        shadow_clip_out,
        bg_change_material_index,
        bg_crest_change_material_index,
        lods,
        meshes,
        attribute_name_offsets,
        submeshes,
        material_name_offsets,
        bone_name_offsets,
        bone_tables,
        shapes,
        shape_meshes,
        shape_values,
        element_id_data,
        extra_lod_data,
        terrain_shadow_mesh_data,
        terrain_shadow_submesh_data,
        submesh_bone_map_data,
        neck_morph_data,
        patch_72_data,
        padding_bytes,
        model_bounding_box_data,
        bone_bounding_box_data,
        furniture_bounding_box_data,
    })
}

fn read_vertex_declarations(
    cursor: &mut Cursor<&[u8]>,
    count: u16,
) -> Result<Vec<VertexDeclaration>> {
    let mut declarations = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let declaration_start = cursor.stream_position()?;
        let mut elements = Vec::new();
        for _ in 0..VERTEX_DECLARATION_SLOTS {
            let stream = cursor.read_u8()?;
            let offset = cursor.read_u8()?;
            let vertex_type_raw = cursor.read_u8()?;
            let usage_raw = cursor.read_u8()?;
            let usage_index = cursor.read_u8()?;
            skip(cursor, 3)?;
            if stream == 0xFF {
                break;
            }
            elements.push(VertexElement {
                stream,
                offset,
                vertex_type: VertexType::try_from(vertex_type_raw)?,
                usage: VertexUsage::try_from(usage_raw)?,
                usage_index,
            });
        }
        cursor.seek(SeekFrom::Start(
            declaration_start + VERTEX_DECLARATION_SLOTS as u64 * VERTEX_ELEMENT_SIZE,
        ))?;
        declarations.push(VertexDeclaration { elements });
    }
    Ok(declarations)
}

fn read_lod(cursor: &mut Cursor<&[u8]>) -> Result<MeshLod> {
    let mesh_index = cursor.read_u16::<LittleEndian>()?;
    let mesh_count = cursor.read_u16::<LittleEndian>()?;
    let model_lod_range = cursor.read_f32::<LittleEndian>()?;
    let texture_lod_range = cursor.read_f32::<LittleEndian>()?;
    let water_mesh_index = cursor.read_u16::<LittleEndian>()?;
    let water_mesh_count = cursor.read_u16::<LittleEndian>()?;
    let shadow_mesh_index = cursor.read_u16::<LittleEndian>()?;
    let shadow_mesh_count = cursor.read_u16::<LittleEndian>()?;
    let terrain_shadow_mesh_index = cursor.read_u16::<LittleEndian>()?;
    let terrain_shadow_mesh_count = cursor.read_u16::<LittleEndian>()?;
    let vertical_fog_mesh_index = cursor.read_u16::<LittleEndian>()?;
    let vertical_fog_mesh_count = cursor.read_u16::<LittleEndian>()?;
    let edge_geometry_size = cursor.read_u32::<LittleEndian>()?;
    let edge_geometry_data_offset = cursor.read_u32::<LittleEndian>()?;
    let unknown6 = cursor.read_u32::<LittleEndian>()?;
    let unknown7 = cursor.read_u32::<LittleEndian>()?;
    let vertex_buffer_size = cursor.read_u32::<LittleEndian>()?;
    let index_buffer_size = cursor.read_u32::<LittleEndian>()?;
    let vertex_data_offset = cursor.read_u32::<LittleEndian>()?;
    let index_data_offset = cursor.read_u32::<LittleEndian>()?;
    Ok(MeshLod {
        mesh_index,
        mesh_count,
        model_lod_range,
        texture_lod_range,
        water_mesh_index,
        water_mesh_count,
        shadow_mesh_index,
        shadow_mesh_count,
        terrain_shadow_mesh_index,
        terrain_shadow_mesh_count,
        vertical_fog_mesh_index,
        vertical_fog_mesh_count,
        edge_geometry_size,
        edge_geometry_data_offset,
        vertex_buffer_size,
        index_buffer_size,
        vertex_data_offset,
        index_data_offset,
        unknown6,
        unknown7,
    })
}

fn read_mesh(cursor: &mut Cursor<&[u8]>) -> Result<Mesh> {
    let vertex_count = cursor.read_u16::<LittleEndian>()?;
    let _padding = cursor.read_u16::<LittleEndian>()?;
    let index_count = cursor.read_u32::<LittleEndian>()?;
    let material_index = cursor.read_u16::<LittleEndian>()?;
    let submesh_index = cursor.read_u16::<LittleEndian>()?;
    let submesh_count = cursor.read_u16::<LittleEndian>()?;
    let bone_table_index = cursor.read_u16::<LittleEndian>()?;
    let start_index = cursor.read_u32::<LittleEndian>()?;
    let vertex_buffer_offsets = read_u32_array_3(cursor)?;
    let vertex_buffer_strides = [cursor.read_u8()?, cursor.read_u8()?, cursor.read_u8()?];
    let vertex_stream_count = cursor.read_u8()?;
    Ok(Mesh {
        vertex_count,
        index_count,
        material_index,
        submesh_index,
        submesh_count,
        bone_table_index,
        start_index,
        vertex_buffer_offsets,
        vertex_buffer_strides,
        vertex_stream_count,
    })
}

fn read_submesh(cursor: &mut Cursor<&[u8]>) -> Result<Submesh> {
    Ok(Submesh {
        index_offset: cursor.read_u32::<LittleEndian>()?,
        index_count: cursor.read_u32::<LittleEndian>()?,
        attribute_index_mask: cursor.read_u32::<LittleEndian>()?,
        bone_start_index: cursor.read_u16::<LittleEndian>()?,
        bone_count: cursor.read_u16::<LittleEndian>()?,
    })
}

fn read_bone_tables(
    cursor: &mut Cursor<&[u8]>,
    version: u32,
    count: u16,
    bone_set_size: u16,
) -> Result<Vec<Vec<u16>>> {
    let mut tables = Vec::with_capacity(count as usize);
    if version <= 0x0100_0005 {
        for _ in 0..count {
            let mut values = Vec::with_capacity(64);
            for _ in 0..64 {
                values.push(cursor.read_u16::<LittleEndian>()?);
            }
            let bone_count = cursor.read_u8()? as usize;
            skip(cursor, 3)?;
            values.truncate(bone_count.min(values.len()));
            tables.push(values);
        }
    } else {
        let block_start = cursor.stream_position()?;
        let block_end = block_start + count as u64 * 4 + bone_set_size as u64 * 2;
        let mut offset_counts = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let offset = cursor.read_u16::<LittleEndian>()?;
            let bone_count = cursor.read_u16::<LittleEndian>()?;
            offset_counts.push((offset, bone_count));
        }
        for (_, bone_count) in offset_counts {
            let mut values = Vec::with_capacity(bone_count as usize);
            for _ in 0..bone_count {
                values.push(cursor.read_u16::<LittleEndian>()?);
            }
            tables.push(values);
            pad_to_4(cursor)?;
        }

        // Version 6 bone-set blocks reserve the full byte range declared by
        // BoneSetSize, even when the counted bone indices only occupy part of
        // it. ConsoleTools advances to the declared block end before reading
        // shapes/part-bone data; doing the same keeps later reads aligned for
        // mods that preserve extra bone-set capacity.
        let current_position = cursor.stream_position()?;
        if current_position < block_end {
            cursor.seek(SeekFrom::Start(block_end))?;
        }
    }
    Ok(tables)
}

fn read_shape_struct(cursor: &mut Cursor<&[u8]>) -> Result<ShapeStruct> {
    Ok(ShapeStruct {
        string_offset: cursor.read_u32::<LittleEndian>()?,
        shape_mesh_start_index: [
            cursor.read_u16::<LittleEndian>()?,
            cursor.read_u16::<LittleEndian>()?,
            cursor.read_u16::<LittleEndian>()?,
        ],
        shape_mesh_count: [
            cursor.read_u16::<LittleEndian>()?,
            cursor.read_u16::<LittleEndian>()?,
            cursor.read_u16::<LittleEndian>()?,
        ],
    })
}

fn read_shape_mesh(cursor: &mut Cursor<&[u8]>) -> Result<ShapeMesh> {
    Ok(ShapeMesh {
        mesh_index_offset: cursor.read_u32::<LittleEndian>()?,
        shape_value_count: cursor.read_u32::<LittleEndian>()?,
        shape_value_offset: cursor.read_u32::<LittleEndian>()?,
    })
}

fn read_shape_value(cursor: &mut Cursor<&[u8]>) -> Result<ShapeValue> {
    Ok(ShapeValue {
        base_indices_index: cursor.read_u16::<LittleEndian>()?,
        replacing_vertex_index: cursor.read_u16::<LittleEndian>()?,
    })
}

fn decode_lod_meshes(bytes: &[u8], data: &ModelData, lod_index: usize) -> Result<Vec<DecodedMesh>> {
    let lod = data
        .lods
        .get(lod_index)
        .with_context(|| format!("missing LoD {lod_index}"))?;
    let mut decoded = Vec::with_capacity(lod.mesh_count as usize);
    for mesh_index in lod.mesh_index..lod.mesh_index + lod.mesh_count {
        let mesh = data
            .meshes
            .get(mesh_index as usize)
            .with_context(|| format!("missing mesh {mesh_index}"))?
            .clone();
        let declaration = data
            .vertex_declarations
            .get(mesh_index as usize)
            .with_context(|| format!("missing vertex declaration for mesh {mesh_index}"))?;
        let vertices = decode_vertices(bytes, lod, &mesh, declaration)?;
        let indices = decode_indices(bytes, lod, &mesh)?;
        let submesh_start = mesh.submesh_index as usize;
        let submesh_end = submesh_start + mesh.submesh_count as usize;
        let submeshes = data
            .submeshes
            .get(submesh_start..submesh_end)
            .with_context(|| format!("invalid submesh range {submesh_start}..{submesh_end}"))?
            .to_vec();
        let bone_names = data
            .bone_tables
            .get(mesh.bone_table_index as usize)
            .map(|table| table.iter().map(|bone| data.bone_name(*bone)).collect())
            .unwrap_or_default();
        decoded.push(DecodedMesh {
            original_mesh_index: mesh_index as usize,
            mesh,
            vertices,
            indices,
            submeshes,
            bone_names,
        });
    }
    Ok(decoded)
}

fn decode_vertices(
    bytes: &[u8],
    lod: &MeshLod,
    mesh: &Mesh,
    declaration: &VertexDeclaration,
) -> Result<Vec<Vertex>> {
    let mut vertices = vec![Vertex::default(); mesh.vertex_count as usize];
    for (vertex_index, vertex) in vertices
        .iter_mut()
        .enumerate()
        .take(mesh.vertex_count as usize)
    {
        for element in &declaration.elements {
            if element.stream as usize >= mesh.vertex_stream_count as usize
                || element.stream as usize >= 3
            {
                continue;
            }
            let offset = lod.vertex_data_offset as u64
                + mesh.vertex_buffer_offsets[element.stream as usize] as u64
                + element.offset as u64
                + mesh.vertex_buffer_strides[element.stream as usize] as u64 * vertex_index as u64;
            let mut cursor = Cursor::new(bytes);
            cursor.seek(SeekFrom::Start(offset))?;
            match element.usage {
                VertexUsage::Position => {
                    vertex.position = read_vec3(&mut cursor, element.vertex_type)?
                }
                VertexUsage::Normal => vertex.normal = read_vec3(&mut cursor, element.vertex_type)?,
                VertexUsage::Binormal => {
                    let (value, handedness) = read_byte_vector3(&mut cursor)?;
                    vertex.binormal = value;
                    vertex.binormal_handedness = handedness;
                }
                VertexUsage::Flow => vertex.flow = read_byte_vector3(&mut cursor)?.0,
                VertexUsage::TextureCoordinate => {
                    let (uv_a, uv_b) = read_uv_pair(&mut cursor, element.vertex_type)?;
                    if element.usage_index == 0 {
                        vertex.uv1 = uv_a;
                        vertex.uv2 = uv_b.unwrap_or([0.0; 2]);
                    } else {
                        vertex.uv3 = uv_a;
                    }
                }
                VertexUsage::Color => {
                    if element.usage_index == 0 {
                        vertex.color = read_color(&mut cursor)?;
                    } else {
                        vertex.color2 = read_color(&mut cursor)?;
                    }
                }
                VertexUsage::BoneWeight => {
                    let weights = read_weight_array(&mut cursor, element.vertex_type)?;
                    for (index, value) in weights.iter().enumerate().take(8) {
                        vertex.bone_weights[index] = *value;
                    }
                }
                VertexUsage::BoneIndex => {
                    let ids = read_byte_array(&mut cursor, element.vertex_type)?;
                    for (index, value) in ids.iter().enumerate().take(8) {
                        vertex.bone_ids[index] = *value;
                    }
                }
            }
        }
    }
    Ok(vertices)
}

fn decode_indices(bytes: &[u8], lod: &MeshLod, mesh: &Mesh) -> Result<Vec<u16>> {
    let offset = lod.index_data_offset as u64 + mesh.start_index as u64 * 2;
    let mut cursor = Cursor::new(bytes);
    cursor.seek(SeekFrom::Start(offset))?;
    let mut indices = Vec::with_capacity(mesh.index_count as usize);
    for _ in 0..mesh.index_count {
        indices.push(cursor.read_u16::<LittleEndian>()?);
    }
    Ok(indices)
}

fn read_vec3(cursor: &mut Cursor<&[u8]>, vertex_type: VertexType) -> Result<[f32; 3]> {
    Ok(match vertex_type {
        VertexType::Float3 => [
            cursor.read_f32::<LittleEndian>()?,
            cursor.read_f32::<LittleEndian>()?,
            cursor.read_f32::<LittleEndian>()?,
        ],
        VertexType::Float4 => [
            cursor.read_f32::<LittleEndian>()?,
            cursor.read_f32::<LittleEndian>()?,
            cursor.read_f32::<LittleEndian>()?,
        ],
        VertexType::Half4 => [read_f16(cursor)?, read_f16(cursor)?, read_f16(cursor)?],
        VertexType::Ubyte4n => read_byte_vector3(cursor)?.0,
        _ => bail!("unsupported vec3 vertex type {:?}", vertex_type),
    })
}

fn read_uv_pair(
    cursor: &mut Cursor<&[u8]>,
    vertex_type: VertexType,
) -> Result<([f32; 2], Option<[f32; 2]>)> {
    Ok(match vertex_type {
        VertexType::Float2 => (
            [
                cursor.read_f32::<LittleEndian>()?,
                cursor.read_f32::<LittleEndian>()?,
            ],
            None,
        ),
        VertexType::Float4 => (
            [
                cursor.read_f32::<LittleEndian>()?,
                cursor.read_f32::<LittleEndian>()?,
            ],
            Some([
                cursor.read_f32::<LittleEndian>()?,
                cursor.read_f32::<LittleEndian>()?,
            ]),
        ),
        VertexType::Half2 => ([read_f16(cursor)?, read_f16(cursor)?], None),
        VertexType::Half4 => (
            [read_f16(cursor)?, read_f16(cursor)?],
            Some([read_f16(cursor)?, read_f16(cursor)?]),
        ),
        VertexType::Ubyte4n => {
            let values = read_byte_float4(cursor)?;
            ([values[0], values[1]], Some([values[2], values[3]]))
        }
        _ => bail!("unsupported UV vertex type {:?}", vertex_type),
    })
}

fn read_color(cursor: &mut Cursor<&[u8]>) -> Result<[f32; 4]> {
    read_byte_float4(cursor)
}

fn read_weight_array(cursor: &mut Cursor<&[u8]>, vertex_type: VertexType) -> Result<Vec<f32>> {
    Ok(read_byte_array(cursor, vertex_type)?
        .into_iter()
        .map(|value| value as f32 / 255.0)
        .collect())
}

fn read_byte_array(cursor: &mut Cursor<&[u8]>, vertex_type: VertexType) -> Result<Vec<u8>> {
    let len = match vertex_type {
        VertexType::Ubyte8 => 8,
        VertexType::Ubyte4 | VertexType::Ubyte4n => 4,
        _ => bail!("unsupported byte-array vertex type {:?}", vertex_type),
    };
    let mut raw = vec![0; len];
    cursor.read_exact(&mut raw)?;
    if vertex_type == VertexType::Ubyte8 {
        Ok(vec![
            raw[0], raw[2], raw[4], raw[6], raw[1], raw[3], raw[5], raw[7],
        ])
    } else {
        Ok(raw)
    }
}

fn read_byte_float4(cursor: &mut Cursor<&[u8]>) -> Result<[f32; 4]> {
    Ok([
        cursor.read_u8()? as f32 / 255.0,
        cursor.read_u8()? as f32 / 255.0,
        cursor.read_u8()? as f32 / 255.0,
        cursor.read_u8()? as f32 / 255.0,
    ])
}

fn read_byte_vector3(cursor: &mut Cursor<&[u8]>) -> Result<([f32; 3], bool)> {
    let x = cursor.read_u8()? as f32 * 2.0 / 255.0 - 1.0;
    let y = cursor.read_u8()? as f32 * 2.0 / 255.0 - 1.0;
    let z = cursor.read_u8()? as f32 * 2.0 / 255.0 - 1.0;
    let handedness = cursor.read_u8()? != 0;
    Ok(([x, y, z], handedness))
}

fn read_f16(cursor: &mut Cursor<&[u8]>) -> Result<f32> {
    Ok(f16::from_bits(cursor.read_u16::<LittleEndian>()?).to_f32())
}

fn read_u32_array_3(cursor: &mut Cursor<&[u8]>) -> Result<[u32; 3]> {
    Ok([
        cursor.read_u32::<LittleEndian>()?,
        cursor.read_u32::<LittleEndian>()?,
        cursor.read_u32::<LittleEndian>()?,
    ])
}

fn read_u32_vec(cursor: &mut Cursor<&[u8]>, count: usize) -> Result<Vec<u32>> {
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(cursor.read_u32::<LittleEndian>()?);
    }
    Ok(values)
}

fn skip(cursor: &mut Cursor<&[u8]>, count: u64) -> Result<()> {
    cursor.seek(SeekFrom::Current(count as i64))?;
    Ok(())
}

fn read_bytes(cursor: &mut Cursor<&[u8]>, count: usize) -> Result<Vec<u8>> {
    let mut bytes = vec![0; count];
    cursor.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn pad_to_4(cursor: &mut Cursor<&[u8]>) -> Result<()> {
    let position = cursor.stream_position()?;
    let padding = (4 - position % 4) % 4;
    skip(cursor, padding)
}

fn read_c_string(bytes: &[u8], offset: usize) -> String {
    if offset >= bytes.len() {
        return String::new();
    }
    let end = bytes[offset..]
        .iter()
        .position(|value| *value == 0)
        .map(|relative| offset + relative)
        .unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[offset..end]).into_owned()
}

fn calculate_tangents_from_binormals(vertices: &mut [Vertex]) {
    for vertex in vertices {
        let sign = if vertex.binormal_handedness {
            -1.0
        } else {
            1.0
        };
        vertex.tangent = mul3(cross3(vertex.normal, vertex.binormal), sign);
    }
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn mul3(a: [f32; 3], value: f32) -> [f32; 3] {
    [a[0] * value, a[1] * value, a[2] * value]
}
