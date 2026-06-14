// SPDX-License-Identifier: GPL-3.0-or-later

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rusqlite::{Connection, params};

use crate::mdl::{DecodedMesh, MeshPartExport, ModelFile, Vertex};

const CREATE_IMPORT_DB_SQL: &str = r#"
CREATE TABLE "meta" (
    "key" TEXT NOT NULL UNIQUE,
    "value" TEXT,
    PRIMARY KEY("key")
);
CREATE TABLE "warnings" ("text" TEXT NOT NULL);
CREATE TABLE "indices" (
    "mesh" INTEGER NOT NULL,
    "part" INTEGER NOT NULL,
    "index_id" INTEGER NOT NULL,
    "vertex_id" INTEGER NOT NULL,
    PRIMARY KEY("mesh","part","index_id")
);
CREATE TABLE "vertices" (
    "mesh" INTEGER NOT NULL,
    "part" INTEGER NOT NULL,
    "vertex_id" INTEGER NOT NULL,
    "position_x" REAL NOT NULL,
    "position_y" REAL NOT NULL,
    "position_z" REAL NOT NULL,
    "normal_x" REAL NOT NULL,
    "normal_y" REAL NOT NULL,
    "normal_z" REAL NOT NULL,
    "binormal_x" REAL,
    "binormal_y" REAL,
    "binormal_z" REAL,
    "tangent_x" REAL,
    "tangent_y" REAL,
    "tangent_z" REAL,
    "color_r" REAL NOT NULL,
    "color_g" REAL NOT NULL,
    "color_b" REAL NOT NULL,
    "color_a" REAL NOT NULL,
    "color2_r" REAL NOT NULL,
    "color2_g" REAL NOT NULL,
    "color2_b" REAL NOT NULL,
    "color2_a" REAL NOT NULL,
    "uv_1_u" REAL NOT NULL,
    "uv_1_v" REAL NOT NULL,
    "uv_2_u" REAL NOT NULL,
    "uv_2_v" REAL NOT NULL,
    "uv_3_u" REAL NOT NULL,
    "uv_3_v" REAL NOT NULL,
    "bone_1_id" INTEGER,
    "bone_1_weight" REAL,
    "bone_2_id" INTEGER,
    "bone_2_weight" REAL,
    "bone_3_id" INTEGER,
    "bone_3_weight" REAL,
    "bone_4_id" INTEGER,
    "bone_4_weight" REAL,
    "bone_5_id" INTEGER,
    "bone_5_weight" REAL,
    "bone_6_id" INTEGER,
    "bone_6_weight" REAL,
    "bone_7_id" INTEGER,
    "bone_7_weight" REAL,
    "bone_8_id" INTEGER,
    "bone_8_weight" REAL,
    "flow_u" REAL NOT NULL,
    "flow_v" REAL NOT NULL,
    PRIMARY KEY("mesh","part","vertex_id")
);
CREATE TABLE "shape_vertices" (
    "shape" TEXT NOT NULL,
    "mesh" INTEGER NOT NULL,
    "part" INTEGER NOT NULL,
    "vertex_id" INTEGER NOT NULL,
    "position_x" REAL NOT NULL,
    "position_y" REAL NOT NULL,
    "position_z" REAL NOT NULL,
    PRIMARY KEY("shape", "mesh", "part", "vertex_id")
);
CREATE TABLE "shape_values" (
    "shape" TEXT NOT NULL,
    "mesh" INTEGER NOT NULL,
    "part" INTEGER NOT NULL,
    "shape_value_index" INTEGER NOT NULL,
    "base_indices_index" INTEGER NOT NULL,
    "replacing_vertex_index" INTEGER NOT NULL,
    "vertex_id" INTEGER NOT NULL,
    "position_x" REAL NOT NULL,
    "position_y" REAL NOT NULL,
    "position_z" REAL NOT NULL,
    PRIMARY KEY("shape", "mesh", "part", "shape_value_index")
);
CREATE TABLE "models" (
    "model" INTEGER NOT NULL,
    "name" TEXT,
    Primary KEY("model")
);
CREATE TABLE "meshes" (
    "mesh" INTEGER NOT NULL,
    "model" INTEGER NOT NULL,
    "material_id" INTEGER,
    "name" TEXT,
    "type" TEXT,
    Primary KEY("mesh")
);
CREATE TABLE "parts" (
    "mesh" INTEGER NOT NULL,
    "part" INTEGER NOT NULL,
    "name" TEXT,
    "attributes" TEXT,
    Primary KEY("mesh", "part")
);
CREATE TABLE "bones" (
    "mesh" INTEGER NOT NULL,
    "bone_id" INTEGER NOT NULL,
    "name" TEXT NOT NULL,
    Primary KEY("mesh","bone_id")
);
CREATE TABLE "skeleton" (
    "name" TEXT NOT NULL,
    "parent" TEXT,
    "matrix_0" REAL,
    "matrix_1" REAL,
    "matrix_2" REAL,
    "matrix_3" REAL,
    "matrix_4" REAL,
    "matrix_5" REAL,
    "matrix_6" REAL,
    "matrix_7" REAL,
    "matrix_8" REAL,
    "matrix_9" REAL,
    "matrix_10" REAL,
    "matrix_11" REAL,
    "matrix_12" REAL,
    "matrix_13" REAL,
    "matrix_14" REAL,
    "matrix_15" REAL,
    Primary KEY("name")
);
CREATE TABLE "materials" (
    "material_id" INTEGER NOT NULL,
    "name" TEXT,
    "diffuse" TEXT,
    "normal" TEXT,
    "specular" TEXT,
    "opacity" TEXT,
    "emissive" TEXT,
    Primary KEY("material_id")
);
"#;

pub fn write_import_db(model: &ModelFile, dest_db: &Path, _game_path: &str) -> Result<()> {
    if let Some(parent) = dest_db.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let temp_path = temp_db_path(dest_db);
    if temp_path.exists() {
        std::fs::remove_file(&temp_path)?;
    }

    let mut conn = Connection::open(&temp_path)
        .with_context(|| format!("failed to open {}", temp_path.display()))?;
    conn.execute_batch(CREATE_IMPORT_DB_SQL)?;
    let tx = conn.transaction()?;

    insert_meta(&tx, model)?;
    tx.execute(
        "INSERT INTO models (model, name) VALUES (?1, ?2)",
        params![0i64, model.source_stem],
    )?;
    let material_id_map = insert_materials(&tx, model, dest_db)?;

    for (mesh_id, mesh) in model.lod0_meshes.iter().enumerate() {
        insert_mesh(&tx, mesh_id, mesh, &material_id_map)?;
        let parts = mesh.export_parts()?;
        insert_bones(&tx, mesh_id, mesh)?;
        for part in &parts {
            insert_part(&tx, model, mesh_id, part)?;
            insert_vertices(&tx, mesh_id, part)?;
            insert_indices(&tx, mesh_id, part)?;
        }
        insert_shape_vertices(&tx, model, mesh_id, mesh, &parts)?;
    }

    tx.commit()?;
    drop(conn);
    if dest_db.exists() {
        std::fs::remove_file(dest_db)?;
    }
    std::fs::rename(&temp_path, dest_db).with_context(|| {
        format!(
            "failed to move {} to {}",
            temp_path.display(),
            dest_db.display()
        )
    })?;
    Ok(())
}

fn insert_meta(conn: &Connection, model: &ModelFile) -> Result<()> {
    let entries = [
        ("unit", "meter"),
        ("application", "ffxiv_mdl_converter"),
        ("version", env!("CARGO_PKG_VERSION")),
        ("up", "y"),
        ("front", "z"),
        ("handedness", "r"),
        ("for_3ds_max", "0"),
        ("root_name", model.source_stem.as_str()),
    ];
    for (key, value) in entries {
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?1, ?2)",
            params![key, value],
        )?;
    }
    Ok(())
}

fn insert_materials(conn: &Connection, model: &ModelFile, dest_db: &Path) -> Result<Vec<usize>> {
    let texture_dir = dest_db.parent().unwrap_or_else(|| Path::new("."));

    let mut ordered_materials = (0..model.data.material_name_offsets.len())
        .map(|raw_index| {
            let raw_name = model.data.material_name(raw_index as u16);
            let material_name = material_stem(&raw_name);
            (raw_index, material_name)
        })
        .collect::<Vec<_>>();
    ordered_materials.sort_by(|left, right| left.1.cmp(&right.1).then(left.0.cmp(&right.0)));

    let mut material_id_map = vec![0usize; ordered_materials.len()];
    for (material_id, (raw_index, material_name)) in ordered_materials.into_iter().enumerate() {
        material_id_map[raw_index] = material_id;
        let prefix = texture_dir.join(format!("{material_name}_"));
        conn.execute(
            "INSERT INTO materials (material_id, name, diffuse, normal, specular, opacity, emissive) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                material_id as i64,
                material_name,
                texture_placeholder(&prefix, "d"),
                texture_placeholder(&prefix, "n"),
                texture_placeholder(&prefix, "s"),
                texture_placeholder(&prefix, "o"),
                texture_placeholder(&prefix, "e"),
            ],
        )?;
    }
    Ok(material_id_map)
}

fn insert_mesh(
    conn: &Connection,
    mesh_id: usize,
    mesh: &DecodedMesh,
    material_id_map: &[usize],
) -> Result<()> {
    let material_id = material_id_map
        .get(mesh.mesh.material_index as usize)
        .copied()
        .unwrap_or(mesh.mesh.material_index as usize);
    conn.execute(
        "INSERT INTO meshes (mesh, model, material_id, name, type) VALUES (?1, ?2, ?3, ?4, ?5)",
        params![
            mesh_id as i64,
            0i64,
            material_id as i64,
            format!("Group {mesh_id}"),
            "Standard",
        ],
    )?;
    Ok(())
}

fn insert_bones(conn: &Connection, mesh_id: usize, mesh: &DecodedMesh) -> Result<()> {
    for (bone_id, bone_name) in mesh.bone_names.iter().enumerate() {
        conn.execute(
            "INSERT INTO bones (mesh, bone_id, name) VALUES (?1, ?2, ?3)",
            params![mesh_id as i64, bone_id as i64, bone_name],
        )?;
    }
    Ok(())
}

fn insert_part(
    conn: &Connection,
    model: &ModelFile,
    mesh_id: usize,
    part: &MeshPartExport,
) -> Result<()> {
    let attrs = model
        .data
        .attributes_for_mask(part.submesh.attribute_index_mask)
        .join(",");
    conn.execute(
        "INSERT INTO parts (mesh, part, name, attributes) VALUES (?1, ?2, ?3, ?4)",
        params![
            mesh_id as i64,
            part.part_index as i64,
            format!("Part {}", part.part_index),
            attrs,
        ],
    )?;
    Ok(())
}

fn insert_vertices(conn: &Connection, mesh_id: usize, part: &MeshPartExport) -> Result<()> {
    let mut stmt = conn.prepare(
        "INSERT INTO vertices (
            mesh, part, vertex_id,
            position_x, position_y, position_z,
            normal_x, normal_y, normal_z,
            binormal_x, binormal_y, binormal_z,
            tangent_x, tangent_y, tangent_z,
            color_r, color_g, color_b, color_a,
            color2_r, color2_g, color2_b, color2_a,
            uv_1_u, uv_1_v, uv_2_u, uv_2_v, uv_3_u, uv_3_v,
            bone_1_id, bone_1_weight, bone_2_id, bone_2_weight,
            bone_3_id, bone_3_weight, bone_4_id, bone_4_weight,
            bone_5_id, bone_5_weight, bone_6_id, bone_6_weight,
            bone_7_id, bone_7_weight, bone_8_id, bone_8_weight,
            flow_u, flow_v
        ) VALUES (
            ?1, ?2, ?3,
            ?4, ?5, ?6,
            ?7, ?8, ?9,
            ?10, ?11, ?12,
            ?13, ?14, ?15,
            ?16, ?17, ?18, ?19,
            ?20, ?21, ?22, ?23,
            ?24, ?25, ?26, ?27, ?28, ?29,
            ?30, ?31, ?32, ?33,
            ?34, ?35, ?36, ?37,
            ?38, ?39, ?40, ?41,
            ?42, ?43, ?44, ?45,
            ?46, ?47
        )",
    )?;
    for (vertex_id, vertex) in part.vertices.iter().enumerate() {
        bind_vertex(&mut stmt, mesh_id, part.part_index, vertex_id, vertex)?;
    }
    Ok(())
}

fn bind_vertex(
    stmt: &mut rusqlite::Statement<'_>,
    mesh_id: usize,
    part_id: usize,
    vertex_id: usize,
    vertex: &Vertex,
) -> Result<()> {
    stmt.execute(params![
        mesh_id as i64,
        part_id as i64,
        vertex_id as i64,
        vertex.position[0],
        vertex.position[1],
        vertex.position[2],
        vertex.normal[0],
        vertex.normal[1],
        vertex.normal[2],
        vertex.binormal[0],
        vertex.binormal[1],
        vertex.binormal[2],
        vertex.tangent[0],
        vertex.tangent[1],
        vertex.tangent[2],
        vertex.color[0],
        vertex.color[1],
        vertex.color[2],
        vertex.color[3],
        vertex.color2[0],
        vertex.color2[1],
        vertex.color2[2],
        vertex.color2[3],
        vertex.uv1[0],
        vertex.uv1[1],
        vertex.uv2[0],
        vertex.uv2[1],
        vertex.uv3[0],
        vertex.uv3[1],
        vertex.bone_ids[0] as i64,
        vertex.bone_weights[0],
        vertex.bone_ids[1] as i64,
        vertex.bone_weights[1],
        vertex.bone_ids[2] as i64,
        vertex.bone_weights[2],
        vertex.bone_ids[3] as i64,
        vertex.bone_weights[3],
        vertex.bone_ids[4] as i64,
        vertex.bone_weights[4],
        vertex.bone_ids[5] as i64,
        vertex.bone_weights[5],
        vertex.bone_ids[6] as i64,
        vertex.bone_weights[6],
        vertex.bone_ids[7] as i64,
        vertex.bone_weights[7],
        vertex.flow[0],
        vertex.flow[1],
    ])?;
    Ok(())
}

fn insert_indices(conn: &Connection, mesh_id: usize, part: &MeshPartExport) -> Result<()> {
    let mut stmt = conn
        .prepare("INSERT INTO indices (mesh, part, index_id, vertex_id) VALUES (?1, ?2, ?3, ?4)")?;
    for (index_id, vertex_id) in part.indices.iter().enumerate() {
        stmt.execute(params![
            mesh_id as i64,
            part.part_index as i64,
            index_id as i64,
            *vertex_id as i64,
        ])?;
    }
    Ok(())
}

fn insert_shape_vertices(
    conn: &Connection,
    model: &ModelFile,
    mesh_id: usize,
    mesh: &DecodedMesh,
    parts: &[MeshPartExport],
) -> Result<()> {
    let mut vertex_stmt = conn.prepare(
        "INSERT OR REPLACE INTO shape_vertices (shape, mesh, part, vertex_id, position_x, position_y, position_z)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
    )?;
    let mut value_stmt = conn.prepare(
        "INSERT OR REPLACE INTO shape_values (
            shape, mesh, part, shape_value_index, base_indices_index, replacing_vertex_index,
            vertex_id, position_x, position_y, position_z
        )
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
    )?;
    for shape in &model.data.shapes {
        let shape_name = model.data.shape_name(shape);
        if !shape_name.starts_with("shp") {
            continue;
        }
        let start = shape.shape_mesh_start_index[0] as usize;
        let count = shape.shape_mesh_count[0] as usize;
        for shape_mesh in model.data.shape_meshes.iter().skip(start).take(count) {
            if shape_mesh.mesh_index_offset != mesh.mesh.start_index {
                continue;
            }
            let value_start = shape_mesh.shape_value_offset as usize;
            let value_count = shape_mesh.shape_value_count as usize;
            for (relative_value_index, value) in model
                .data
                .shape_values
                .iter()
                .skip(value_start)
                .take(value_count)
                .enumerate()
            {
                let absolute_index_position = value
                    .base_indices_index
                    .checked_sub(mesh.mesh.start_index as u16)
                    .map(|value| value as usize);
                let index_position = absolute_index_position
                    .filter(|position| *position < mesh.indices.len())
                    .or_else(|| {
                        let relative_position = value.base_indices_index as usize;
                        (relative_position < mesh.indices.len()).then_some(relative_position)
                    });
                let Some(index_position) = index_position else {
                    continue;
                };
                let Some(original_vertex_id) = mesh.indices.get(index_position).copied() else {
                    continue;
                };
                let Some(replacement) = mesh
                    .vertices
                    .get(value.replacing_vertex_index as usize)
                    .copied()
                else {
                    continue;
                };
                let Some(part) =
                    part_for_index_position(parts, mesh.mesh.start_index, index_position)
                else {
                    continue;
                };
                let Some(part_vertex_id) = part.original_to_part_vertex.get(&original_vertex_id)
                else {
                    continue;
                };
                vertex_stmt.execute(params![
                    shape_name,
                    mesh_id as i64,
                    part.part_index as i64,
                    *part_vertex_id as i64,
                    replacement.position[0],
                    replacement.position[1],
                    replacement.position[2],
                ])?;
                value_stmt.execute(params![
                    shape_name,
                    mesh_id as i64,
                    part.part_index as i64,
                    value_start as i64 + relative_value_index as i64,
                    value.base_indices_index as i64,
                    value.replacing_vertex_index as i64,
                    *part_vertex_id as i64,
                    replacement.position[0],
                    replacement.position[1],
                    replacement.position[2],
                ])?;
            }
        }
    }
    Ok(())
}

fn part_for_index_position(
    parts: &[MeshPartExport],
    mesh_start_index: u32,
    index_position: usize,
) -> Option<&MeshPartExport> {
    let absolute_index_position = mesh_start_index as usize + index_position;
    parts.iter().find(|part| {
        let start = part.submesh.index_offset as usize;
        let end = start + part.submesh.index_count as usize;
        absolute_index_position >= start && absolute_index_position < end
    })
}

fn material_stem(path: &str) -> String {
    let normalized = path.trim_start_matches('/').replace('\\', "/");
    let filename = normalized.rsplit('/').next().unwrap_or(normalized.as_str());
    filename
        .strip_suffix(".mtrl")
        .unwrap_or(filename)
        .to_string()
}

fn texture_placeholder(prefix: &Path, suffix: &str) -> String {
    let mut value = PathBuf::from(prefix);
    let filename = format!(
        "{}{}.png",
        value
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("material_"),
        suffix
    );
    value.set_file_name(filename);
    value.to_string_lossy().into_owned()
}

fn temp_db_path(dest_db: &Path) -> PathBuf {
    let file_name = dest_db
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("output.db");
    dest_db.with_file_name(format!(".{file_name}.tmp"))
}
