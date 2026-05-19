// SPDX-License-Identifier: GPL-3.0-or-later

use std::path::Path;

use anyhow::Result;

use crate::mdl::parse_mdl;
use crate::sqlite::write_import_db;

pub fn unwrap_mdl_to_db(source_mdl: &Path, dest_db: &Path, game_path: &str) -> Result<()> {
    let model = parse_mdl(source_mdl)?;
    write_import_db(&model, dest_db, game_path)
}
