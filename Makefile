# PRISM-3D Makefile
# Build, test, and deploy the 3D PDR code

.PHONY: install test run clean container slurm train data help
.PHONY: rust rust-test rust-clean install-rust

PYTHON ?= python3
N ?= 16
G0 ?= 500
OUTPUT ?= ./results
CONTAINER ?= prism3d.sif
TRAIN_SAMPLES ?= 50000
BIND_SRC ?= $(CURDIR)

# Ensure Rust toolchain is in PATH (rustup default location)
export PATH := $(HOME)/.cargo/bin:$(PATH)

help:
	@echo "PRISM-3D v0.6 — 3D PDR with THEMIS Dust Evolution + Rust Core"
	@echo ""
	@echo "Usage:"
	@echo "  make install         Install PRISM-3D with Rust core (maturin)"
	@echo "  make install-python  Install Python-only (no Rust)"
	@echo "  make rust            Build Rust core (release)"
	@echo "  make rust-test       Run Rust unit tests"
	@echo "  make test            Quick validation test (8³)"
	@echo "  make test-parity     Run Rust vs Python parity tests"
	@echo "  make run             Run model (N=$(N), G0=$(G0))"
	@echo "  make run-orion       Run Orion Bar preset"
	@echo "  make run-horsehead   Run Horsehead preset"
	@echo "  make train           Train ML accelerator ($(TRAIN_SAMPLES) samples)"
	@echo "  make data            Download PDRs4All Orion Bar data"
	@echo "  make container       Build Apptainer/Singularity container (with Rust)"
	@echo "  make container-run   Run model inside container"
	@echo "  make container-run-dev  Run with bind-mounted source (development)"
	@echo "  make slurm           Generate SLURM submission scripts"
	@echo "  make clean           Remove build artifacts"
	@echo ""
	@echo "Variables:"
	@echo "  N=32                 Grid cells per dimension"
	@echo "  G0=1000              External FUV field [Habing]"
	@echo "  OUTPUT=./results     Output directory"
	@echo "  TRAIN_SAMPLES=50000  ML training samples"

# ── Installation ──────────────────────────────────────────

install: rust
	@echo "PRISM-3D installed with Rust core"

rust:
	maturin develop --release

install-python:
	@echo "Installing Python-only (no Rust core)..."
	$(PYTHON) -m pip install -e . --no-build-isolation 2>/dev/null || \
	$(PYTHON) -m pip install numpy scipy matplotlib scikit-learn && \
	echo "Python-only install complete (Rust core not available)"

install-hpc: rust
	$(PYTHON) -m pip install mpi4py h5py cupy-cuda12x

install-all: rust
	$(PYTHON) -m pip install mpi4py h5py astropy

# ── Rust ─────────────────────────────────────────────────

rust-test:
	cargo test --manifest-path rust/Cargo.toml

rust-clean:
	cargo clean --manifest-path Cargo.toml

# ── Running ───────────────────────────────────────────────

test:
	$(PYTHON) -m prism3d.run --n 8 --G0 100 --output ./test_output
	@echo "Test complete. Check ./test_output/"

test-parity:
	$(PYTHON) -m pytest prism3d/tests/test_rust_parity.py -v -s

run:
	$(PYTHON) -m prism3d.run --n $(N) --G0 $(G0) --output $(OUTPUT)

run-orion:
	$(PYTHON) -m prism3d.run --orion-bar --n $(N) --output $(OUTPUT)/orion_bar

run-horsehead:
	$(PYTHON) -m prism3d.run --horsehead --n $(N) --output $(OUTPUT)/horsehead

run-config:
	$(PYTHON) -m prism3d.run --config $(CONFIG) --output $(OUTPUT)

# ── ML Accelerator ────────────────────────────────────────

train:
	@echo "Generating $(TRAIN_SAMPLES) BDF training samples..."
	$(PYTHON) -m prism3d.chemistry.train_hpc \
		--n-samples $(TRAIN_SAMPLES) \
		--output training_data_$(TRAIN_SAMPLES).npz
	@echo "Training accelerator..."
	$(PYTHON) -m prism3d.chemistry.train_hpc \
		--train \
		--data training_data_$(TRAIN_SAMPLES).npz \
		--save accelerator_$(TRAIN_SAMPLES).pkl
	@echo "Done. Accelerator: accelerator_$(TRAIN_SAMPLES).pkl"

# ── Data ──────────────────────────────────────────────────

data:
	$(PYTHON) -m prism3d.data.download_data --templates --output ./orion_bar_data

data-full:
	$(PYTHON) -m prism3d.data.download_data --all --output ./orion_bar_data

data-prepare:
	$(PYTHON) -m prism3d.data.prepare_data \
		--input ./orion_bar_data --output ./prepared --n-pix $(N)

# ── Container ─────────────────────────────────────────────

container:
	apptainer build --force --fakeroot $(CONTAINER) apptainer.def

container-run:
	@mkdir -p $(OUTPUT)
	apptainer run --cleanenv --no-mount home --bind $$(realpath $(OUTPUT)):/output \
		$(CONTAINER) --n $(N) --G0 $(G0) --output /output

container-run-dev:
	@mkdir -p $(OUTPUT)
	apptainer run --cleanenv --no-mount home \
		--bind $(BIND_SRC)/prism3d:/opt/prism3d/prism3d \
		--bind $$(realpath $(OUTPUT)):/output \
		$(CONTAINER) --n $(N) --G0 $(G0) --output /output

container-shell:
	apptainer shell --cleanenv --no-mount home $(CONTAINER)

container-train:
	apptainer exec --cleanenv --no-mount home --bind $(CURDIR):/work \
		$(CONTAINER) python -m prism3d.chemistry.train_hpc \
		--n-samples $(TRAIN_SAMPLES) --output /work/training_data_$(TRAIN_SAMPLES).npz
	apptainer exec --cleanenv --no-mount home --bind $(CURDIR):/work \
		$(CONTAINER) python -m prism3d.chemistry.train_hpc \
		--train --data /work/training_data_$(TRAIN_SAMPLES).npz \
		--save /work/accelerator_$(TRAIN_SAMPLES).pkl

# ── SLURM ─────────────────────────────────────────────────

slurm:
	$(PYTHON) -m prism3d.run --gen-slurm --n $(N) --G0 $(G0) \
		--nodes 1 --partition gpu --wall-time 04:00:00
	@echo "Submit with: sbatch run_prism3d_$(N)cube.slurm"

slurm-train:
	$(PYTHON) -m prism3d.chemistry.train_hpc --gen-slurm \
		--n-samples $(TRAIN_SAMPLES) --cpus 16 --wall-time 08:00:00
	@echo "Submit with: sbatch train_accelerator_$(TRAIN_SAMPLES).slurm"

# ── Clean ─────────────────────────────────────────────────

clean:
	rm -rf build/ dist/ *.egg-info __pycache__
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

clean-all: clean rust-clean
	rm -rf test_output/ results/ orion_bar_data/ prepared/
	rm -f *.slurm *.pkl training_data_*.npz
