# ffxiv_mdl_converter

Standalone GPL-3.0-or-later Rust CLI for converting Final Fantasy XIV MDL files to and from the SQLite import database format used by TexTools-compatible tooling.

This project is intentionally self-contained. It exposes only a command-line interface and does not depend on any downstream application, mod-processing pipeline, or application-specific assets.

## License

This project is licensed under GPL-3.0-or-later. See [`COPYING`](COPYING).

## Commands

```bash
ffxiv_mdl_converter unwrap <source.mdl> <dest.db> <game_path>
ffxiv_mdl_converter wrap <source.db> <dest.mdl> <game_path> --base-mdl <source.mdl> [wrap flags]
```

`unwrap` converts an uncompressed FFXIV MDL into a TexTools-compatible SQLite import DB.

`wrap` converts that SQLite import DB back into an uncompressed FFXIV MDL. It uses a base MDL for structural data that is not represented in the DB. Pass `--base-mdl`, or set one of these environment variables to an extracted game-file mirror so the converter can resolve `<game_path>`:

- `FFXIV_MDL_BASE_DIRS` - path-list of extracted roots.
- `FFXIV_MDL_BASE_DIR` - single extracted root.
- `FFXIV_GAMEFILES_EXTRACTED` - single extracted root.
- `FFXIV_GAMEFILES_DIR` - single extracted root.

Supported ConsoleTools-compatible wrap flags:

- `--mats` disables copying material assignments from the base MDL.
- `--attributes` disables copying part attributes/mesh type metadata from the base MDL.
- `--autoscale` disables unit autoscale.
- `--shiftuvs` disables UV V-coordinate shifting.
- `--cloneuv2` copies UV1 into UV2 during import.
- `--heels` disables automatic `heels_offset=` attribute assignment for top/dwn/sho models.
- `--tangents` is accepted for CLI parity. FFXIV MDLs store binormal/handedness rather than a tangent stream, so the converter preserves DB normal/binormal data and uses tangent data only for handedness inference.

## Build

```bash
cargo build --release
```

The release binary is written to:

```text
target/release/ffxiv_mdl_converter
```

Optional install into a destination directory:

```bash
install -Dm755 target/release/ffxiv_mdl_converter ~/.local/bin/ffxiv_mdl_converter
```

## Test

Fast tests do not require game assets:

```bash
cargo test
cargo clippy -- -D warnings
cargo fmt --check
```

Optional fixture-backed tests can exercise real MDL data without committing proprietary or third-party assets to this repository:

```bash
export FFXIV_MDL_CONVERTER_FIXTURE_MDL=/path/to/source.mdl
export FFXIV_MDL_CONVERTER_FIXTURE_GAME_PATH=chara/equipment/e0000/model/c0201e0000_top.mdl
cargo test
```

Optional unwrap parity against a pre-generated SQLite DB:

```bash
export FFXIV_MDL_CONVERTER_GOLDEN_DB=/path/to/golden.db
cargo test unwrap_matches_golden_db_when_fixture_configured
```

Optional live ConsoleTools wrap parity test:

```bash
export FFXIV_CONSOLETOOLS_EXE=/path/to/ConsoleTools.exe
cargo test --test external_fixture -- --ignored
```

## Notes

- Inputs should be uncompressed `.mdl` files.
- The converter intentionally performs no Penumbra, PMP, or mod-package processing. Use external tooling to extract package contents and provide the specific MDL path to convert.
- `.ttmp2`, `.tex`, FBX, and game-file extraction are out of scope for this CLI.
