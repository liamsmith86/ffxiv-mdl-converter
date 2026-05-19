.PHONY: build test lint fmt check clean

build:
	cargo build --release

test:
	cargo test

lint:
	cargo clippy -- -D warnings

fmt:
	cargo fmt

check:
	cargo fmt --check
	cargo clippy -- -D warnings
	cargo test

clean:
	cargo clean
