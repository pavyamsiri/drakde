default: build run

build:
    uv run maturin develop --release

run:
    uv run python -m drakde
