// SPDX-License-Identifier: GPL-3.0-or-later

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use ffxiv_mdl_converter::{unwrap, wrap};

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Convert an uncompressed FFXIV MDL into a TexTools-compatible SQLite import DB.
    Unwrap {
        source_mdl: PathBuf,
        dest_db: PathBuf,
        game_path: String,
    },
    /// Convert a TexTools-compatible SQLite import DB into an uncompressed FFXIV MDL.
    Wrap {
        source_db: PathBuf,
        dest_mdl: PathBuf,
        game_path: String,
        #[arg(long, help = "Source MDL to preserve structural data from during wrap")]
        base_mdl: Option<PathBuf>,
        #[arg(
            long,
            default_value_t = false,
            help = "Match ConsoleTools /mats behavior"
        )]
        mats: bool,
        #[arg(
            long,
            default_value_t = false,
            help = "Match ConsoleTools /attributes behavior"
        )]
        attributes: bool,
        #[arg(
            long,
            default_value_t = false,
            help = "Match ConsoleTools /autoscale behavior"
        )]
        autoscale: bool,
        #[arg(
            long,
            default_value_t = false,
            help = "Match ConsoleTools /shiftuvs behavior"
        )]
        shiftuvs: bool,
        #[arg(
            long,
            default_value_t = false,
            help = "Match ConsoleTools /cloneuv2 behavior"
        )]
        cloneuv2: bool,
        #[arg(
            long,
            default_value_t = false,
            help = "Match ConsoleTools /heels behavior"
        )]
        heels: bool,
        #[arg(
            long,
            default_value_t = false,
            help = "Match ConsoleTools /tangents behavior"
        )]
        tangents: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Unwrap {
            source_mdl,
            dest_db,
            game_path,
        } => unwrap::unwrap_mdl_to_db(&source_mdl, &dest_db, &game_path),
        Command::Wrap {
            source_db,
            dest_mdl,
            game_path,
            base_mdl,
            mats,
            attributes,
            autoscale,
            shiftuvs,
            cloneuv2,
            heels,
            tangents,
        } => wrap::wrap_db_to_mdl(
            &source_db,
            &dest_mdl,
            &game_path,
            &wrap::WrapOptions {
                base_mdl,
                copy_materials: !mats,
                copy_attributes: !attributes,
                autoscale: !autoscale,
                shift_import_uv: !shiftuvs,
                clone_uv2: cloneuv2,
                use_imported_tangents: tangents,
                auto_assign_heels: !heels,
            },
        ),
    }
}
