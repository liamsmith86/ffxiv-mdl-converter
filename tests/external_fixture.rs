// SPDX-License-Identifier: GPL-3.0-or-later

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use rusqlite::Connection;

use ffxiv_mdl_converter::unwrap::unwrap_mdl_to_db;
use ffxiv_mdl_converter::wrap::{WrapOptions, wrap_db_to_mdl};

const COUNT_TABLES: &[&str] = &[
    "materials",
    "meshes",
    "parts",
    "bones",
    "vertices",
    "indices",
    "shape_vertices",
];

const EXACT_TABLES: &[&str] = &[
    "SELECT material_id, name FROM materials ORDER BY material_id",
    "SELECT mesh, model, material_id, name, type FROM meshes ORDER BY mesh",
    "SELECT mesh, bone_id, name FROM bones ORDER BY mesh, bone_id",
    "SELECT mesh, part, index_id, vertex_id FROM indices ORDER BY mesh, part, index_id",
];

const OPTIONAL_EXACT_TABLES: &[(&str, &str)] = &[(
    "shape_values",
    "SELECT shape, mesh, part, shape_value_index, base_indices_index, replacing_vertex_index, vertex_id, position_x, position_y, position_z FROM shape_values ORDER BY shape, mesh, part, shape_value_index",
)];

#[test]
fn roundtrip_fixture_when_configured() -> Result<()> {
    let Some(fixture) = Fixture::from_env()? else {
        eprintln!(
            "skipping external fixture roundtrip; set FFXIV_MDL_CONVERTER_FIXTURE_MDL and FFXIV_MDL_CONVERTER_FIXTURE_GAME_PATH"
        );
        return Ok(());
    };

    let temp = tempfile::tempdir()?;
    let db = temp.path().join("source.db");
    let wrapped = temp.path().join("wrapped.mdl");
    let rewrapped_db = temp.path().join("rewrapped.db");

    unwrap_mdl_to_db(&fixture.source_mdl, &db, &fixture.game_path)?;
    wrap_db_to_mdl(
        &db,
        &wrapped,
        &fixture.game_path,
        &WrapOptions {
            base_mdl: Some(fixture.source_mdl),
            copy_materials: false,
            copy_attributes: false,
            autoscale: false,
            shift_import_uv: true,
            clone_uv2: false,
            use_imported_tangents: false,
            auto_assign_heels: false,
        },
    )?;
    unwrap_mdl_to_db(&wrapped, &rewrapped_db, &fixture.game_path)?;
    assert_db_equivalent(&db, &rewrapped_db)
}

#[test]
fn unwrap_matches_golden_db_when_fixture_configured() -> Result<()> {
    let Some(fixture) = Fixture::from_env()? else {
        eprintln!("skipping external fixture unwrap parity; set fixture env vars");
        return Ok(());
    };
    let Some(golden_db) = optional_path_env("FFXIV_MDL_CONVERTER_GOLDEN_DB")? else {
        eprintln!("skipping external fixture unwrap parity; set FFXIV_MDL_CONVERTER_GOLDEN_DB");
        return Ok(());
    };

    let temp = tempfile::tempdir()?;
    let actual = temp.path().join("actual.db");
    unwrap_mdl_to_db(&fixture.source_mdl, &actual, &fixture.game_path)?;
    assert_db_equivalent(&golden_db, &actual)
}

#[test]
#[ignore = "requires Wine/ConsoleTools and external fixture env vars"]
fn wrap_matches_consoletools_when_fixture_configured() -> Result<()> {
    let fixture = Fixture::from_env()?
        .context("set FFXIV_MDL_CONVERTER_FIXTURE_MDL and FFXIV_MDL_CONVERTER_FIXTURE_GAME_PATH")?;
    let consoletools = required_path_env("FFXIV_CONSOLETOOLS_EXE")?;

    let temp = tempfile::tempdir()?;
    let import_db = temp.path().join("source.db");
    let rust_mdl = temp.path().join("rust.mdl");
    let consoletools_mdl = temp.path().join("consoletools.mdl");
    let rust_db = temp.path().join("rust.db");
    let consoletools_db = temp.path().join("consoletools.db");

    unwrap_mdl_to_db(&fixture.source_mdl, &import_db, &fixture.game_path)?;
    run_consoletools_wrap(
        &consoletools,
        &import_db,
        &consoletools_mdl,
        &fixture.game_path,
    )?;
    wrap_db_to_mdl(
        &import_db,
        &rust_mdl,
        &fixture.game_path,
        &WrapOptions {
            base_mdl: Some(fixture.source_mdl),
            copy_materials: false,
            copy_attributes: false,
            autoscale: false,
            shift_import_uv: true,
            clone_uv2: false,
            use_imported_tangents: false,
            auto_assign_heels: false,
        },
    )?;

    unwrap_mdl_to_db(&consoletools_mdl, &consoletools_db, &fixture.game_path)?;
    unwrap_mdl_to_db(&rust_mdl, &rust_db, &fixture.game_path)?;
    assert_db_equivalent(&consoletools_db, &rust_db)
}

#[derive(Debug)]
struct Fixture {
    source_mdl: PathBuf,
    game_path: String,
}

impl Fixture {
    fn from_env() -> Result<Option<Self>> {
        let Some(source_mdl) = optional_path_env("FFXIV_MDL_CONVERTER_FIXTURE_MDL")? else {
            return Ok(None);
        };
        let Some(game_path) = optional_string_env("FFXIV_MDL_CONVERTER_FIXTURE_GAME_PATH") else {
            return Ok(None);
        };
        Ok(Some(Self {
            source_mdl,
            game_path,
        }))
    }
}

fn optional_string_env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

fn optional_path_env(name: &str) -> Result<Option<PathBuf>> {
    let Some(raw) = optional_string_env(name) else {
        return Ok(None);
    };
    let path = PathBuf::from(raw);
    if !path.exists() {
        bail!("{name} does not exist: {}", path.display());
    }
    Ok(Some(path))
}

fn required_path_env(name: &str) -> Result<PathBuf> {
    optional_path_env(name)?.with_context(|| format!("set {name}"))
}

fn run_consoletools_wrap(
    consoletools: &Path,
    import_db: &Path,
    output_mdl: &Path,
    game_path: &str,
) -> Result<()> {
    let wine = std::env::var_os("WINE").unwrap_or_else(|| "wine".into());
    let output = Command::new(wine)
        .current_dir(consoletools.parent().unwrap_or_else(|| Path::new(".")))
        .arg(consoletools)
        .arg("/wrap")
        .arg(to_wine_path(import_db))
        .arg(to_wine_path(output_mdl))
        .arg(game_path)
        .arg("/mats")
        .arg("/attributes")
        .arg("/autoscale")
        .output()
        .context("failed to spawn ConsoleTools /wrap")?;
    if !output.status.success() {
        bail!(
            "ConsoleTools /wrap failed with {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}

fn to_wine_path(path: &Path) -> String {
    let normalized = path.to_string_lossy().replace('/', "\\");
    if normalized.starts_with('\\') {
        format!("Z:{normalized}")
    } else {
        normalized
    }
}

fn assert_db_equivalent(expected: &Path, actual: &Path) -> Result<()> {
    let expected = Connection::open(expected)?;
    let actual = Connection::open(actual)?;

    for table in COUNT_TABLES {
        let expected_count: i64 =
            expected.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                row.get(0)
            })?;
        let actual_count: i64 =
            actual.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                row.get(0)
            })?;
        assert_eq!(expected_count, actual_count, "count mismatch for {table}");
    }

    for query in EXACT_TABLES {
        let expected_rows = query_rows_as_strings(&expected, query)?;
        let actual_rows = query_rows_as_strings(&actual, query)?;
        assert_eq!(
            expected_rows, actual_rows,
            "exact table mismatch for {query}"
        );
    }
    for (table, query) in OPTIONAL_EXACT_TABLES {
        if table_exists(&expected, table)? && table_exists(&actual, table)? {
            let expected_rows = query_rows_as_strings(&expected, query)?;
            let actual_rows = query_rows_as_strings(&actual, query)?;
            assert_eq!(
                expected_rows, actual_rows,
                "exact optional table mismatch for {query}"
            );
        }
    }

    assert_vertex_positions_close(&expected, &actual)
}

fn table_exists(conn: &Connection, table: &str) -> Result<bool> {
    let mut stmt =
        conn.prepare("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1 LIMIT 1")?;
    let mut rows = stmt.query([table])?;
    Ok(rows.next()?.is_some())
}

fn query_rows_as_strings(conn: &Connection, query: &str) -> Result<Vec<Vec<String>>> {
    let mut stmt = conn.prepare(query)?;
    let column_count = stmt.column_count();
    let rows = stmt.query_map([], |row| {
        let mut values = Vec::with_capacity(column_count);
        for index in 0..column_count {
            values.push(
                row.get::<_, String>(index)
                    .or_else(|_| row.get::<_, i64>(index).map(|value| value.to_string()))
                    .or_else(|_| row.get::<_, f64>(index).map(|value| format!("{value:.8}")))?,
            );
        }
        Ok(values)
    })?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(Into::into)
}

fn assert_vertex_positions_close(expected: &Connection, actual: &Connection) -> Result<()> {
    let query = "SELECT mesh, part, vertex_id, position_x, position_y, position_z FROM vertices ORDER BY mesh, part, vertex_id";
    let expected_rows = vertex_position_rows(expected, query)?;
    let actual_rows = vertex_position_rows(actual, query)?;
    assert_eq!(expected_rows.len(), actual_rows.len(), "vertex row count");
    for (left, right) in expected_rows.iter().zip(actual_rows.iter()) {
        assert_eq!(left.key, right.key, "vertex key");
        for (l, r) in left.position.iter().zip(right.position.iter()) {
            assert!((l - r).abs() < 1e-4, "vertex position mismatch: {l} vs {r}");
        }
    }
    Ok(())
}

#[derive(Debug)]
struct VertexPositionRow {
    key: (i64, i64, i64),
    position: [f64; 3],
}

fn vertex_position_rows(conn: &Connection, query: &str) -> Result<Vec<VertexPositionRow>> {
    let mut stmt = conn.prepare(query)?;
    let rows = stmt.query_map([], |row| {
        Ok(VertexPositionRow {
            key: (row.get(0)?, row.get(1)?, row.get(2)?),
            position: [row.get(3)?, row.get(4)?, row.get(5)?],
        })
    })?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(Into::into)
}
