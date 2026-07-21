default: build run

build:
    uv run maturin develop

run:
    uv run python -m drakde
