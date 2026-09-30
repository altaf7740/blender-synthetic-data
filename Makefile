# Synthetic-data pipeline: CAD parts (STEP) -> instance-segmentation
# training data -> trained YOLO26-seg model.
#
# Every pipeline target goes through src/main.py, which runs each stage in
# the uv venv, in its own process.
#
# Written to run unchanged on macOS/Linux sh and under both shells GNU Make
# may use on Windows: cmd.exe (from PowerShell/cmd) and sh (from Git Bash).
# That means no shell built-ins like rm, and no quotes/parentheses in echo.
#
# Override variables on the command line, e.g.:
#   make all INPUT_DIR=my_parts OUTPUT_DIR=out
#   make generate NUM_FRAMES=20 TEXTURES_DIR=my_photos

INPUT_DIR ?= examples/fasteners
OUTPUT_DIR ?= workspace
NUM_FRAMES ?=
TEXTURES_DIR ?=
PER_CLASS ?= 5
PORT ?= 8000

RUN := uv run python src/main.py --input-dir $(INPUT_DIR) --output-dir $(OUTPUT_DIR) \
	$(if $(NUM_FRAMES),--num-frames $(NUM_FRAMES)) $(if $(TEXTURES_DIR),--textures-dir $(TEXTURES_DIR))

.PHONY: help setup convert generate dataset train all preview ui clean

help:
	@echo Targets: setup convert generate dataset train all preview ui clean
	@echo Vars: INPUT_DIR=$(INPUT_DIR) OUTPUT_DIR=$(OUTPUT_DIR) NUM_FRAMES=$(NUM_FRAMES) TEXTURES_DIR=$(TEXTURES_DIR) PER_CLASS=$(PER_CLASS) PORT=$(PORT)

setup:
	uv sync

convert:
	$(RUN) --stage convert

generate:
	$(RUN) --stage generate

dataset:
	$(RUN) --stage dataset

train:
	$(RUN) --stage train

all:
	$(RUN)

preview:
	uv run python src/preview_dataset.py --output-dir $(OUTPUT_DIR) --per-class $(PER_CLASS)

ui:
	uv run python src/server.py --port $(PORT)

clean:
	uv run python -c "import shutil; shutil.rmtree(r'$(OUTPUT_DIR)', ignore_errors=True)"
