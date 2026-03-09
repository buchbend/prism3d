# PRISM-3D Makefile
# Build, test, and deploy the 3D PDR code

.PHONY: install test run clean container slurm train train-container data help

PYTHON ?= python3
N ?= 16
G0 ?= 500
N_MEAN ?= 500
BOX ?= 2.0
OUTPUT ?= ./results
MAX_ITER ?= 20
DUST_STEPS ?= 5
CONTAINER ?= ../prism3d.sif
ACCELERATOR ?=
TRAIN_SAMPLES ?= 50000
TRAIN_CPUS ?= 16
TRAIN_WALLTIME ?= 08:00:00

help:
	@echo "PRISM-3D v0.5 — 3D PDR with THEMIS Dust Evolution"
	@echo ""
	@echo "Usage:"
	@echo "  make install         Install PRISM-3D (editable)"
	@echo "  make test            Quick validation test (8³)"
	@echo "  make run             Run model (N=$(N), G0=$(G0))"
	@echo "  make run-orion       Run Orion Bar preset"
	@echo "  make run-horsehead   Run Horsehead preset"
	@echo "  make container-run   Run inside Apptainer container"
	@echo "  make container-train Train ML accelerator in container"
	@echo "  make train           Train ML accelerator locally"
	@echo "  make data            Download PDRs4All Orion Bar data"
	@echo "  make container       Build Apptainer/Singularity container"
	@echo "  make slurm           Generate SLURM submission scripts"
	@echo "  make clean           Remove build artifacts"
	@echo ""
	@echo "Variables:"
	@echo "  N=32                 Grid cells per dimension"
	@echo "  G0=1000              External FUV field [Habing]"
	@echo "  OUTPUT=./results     Output directory"
	@echo "  ACCELERATOR=         Path to trained accelerator .pkl"
	@echo "  TRAIN_SAMPLES=50000  ML training samples"
	@echo "  TRAIN_CPUS=16        Worker cores for training"
	@echo "  TRAIN_WALLTIME=08:00:00  SLURM wall time"

# ── Installation ──────────────────────────────────────────

install:
	$(PYTHON) -m pip install -e .

install-hpc:
	$(PYTHON) -m pip install -e ".[hpc]"

install-all:
	$(PYTHON) -m pip install -e ".[all]"

# ── Running ───────────────────────────────────────────────

test:
	$(PYTHON) -m prism3d.run --n 8 --G0 100 --output ./test_output
	@echo "Test complete. Check ./test_output/"

run:
	$(PYTHON) -m prism3d.run --n $(N) --G0 $(G0) --n-mean $(N_MEAN) \
		--box $(BOX) --max-iter $(MAX_ITER) \
		--dust-steps $(DUST_STEPS) --output $(OUTPUT) \
		$(if $(ACCELERATOR),--accelerator $(ACCELERATOR))

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

# Bind-mount local source so the container uses current code,
# and bind output + accelerator into /output.
BIND_SRC = --bind $(CURDIR):/opt/prism3d/prism3d
BIND_OUT = --bind $(abspath $(OUTPUT)):/output
BIND_ACC = $(if $(ACCELERATOR),--bind $(abspath $(ACCELERATOR)):/model/accelerator.pkl)
ACC_FLAG = $(if $(ACCELERATOR),--accelerator /model/accelerator.pkl)

container:
	apptainer build $(CONTAINER) apptainer.def

container-run:
	@mkdir -p $(OUTPUT)
	apptainer exec $(BIND_SRC) $(BIND_OUT) $(BIND_ACC) $(CONTAINER) \
		python -m prism3d.run --n $(N) --G0 $(G0) --n-mean $(N_MEAN) \
		--box $(BOX) --max-iter $(MAX_ITER) \
		--dust-steps $(DUST_STEPS) --output /output $(ACC_FLAG)

container-train:
	@mkdir -p $(OUTPUT)
	apptainer exec $(BIND_SRC) $(BIND_OUT) $(CONTAINER) \
		python -m prism3d.chemistry.train_hpc \
		--n-samples $(TRAIN_SAMPLES) --n-workers $$(nproc) \
		--output /output/training_data_$(TRAIN_SAMPLES).npz
	apptainer exec $(BIND_SRC) $(BIND_OUT) $(CONTAINER) \
		python -m prism3d.chemistry.train_hpc \
		--train --data /output/training_data_$(TRAIN_SAMPLES).npz \
		--save /output/accelerator_$(TRAIN_SAMPLES).pkl
	@echo "Accelerator: $(OUTPUT)/accelerator_$(TRAIN_SAMPLES).pkl"

container-shell:
	apptainer shell $(BIND_SRC) $(CONTAINER)

# ── SLURM ─────────────────────────────────────────────────

slurm:
	$(PYTHON) -m prism3d.run --gen-slurm --n $(N) --G0 $(G0) \
		--nodes 1 --partition gpu --wall-time 04:00:00
	@echo "Submit with: sbatch run_prism3d_$(N)cube.slurm"

slurm-train:
	$(PYTHON) -m prism3d.chemistry.train_hpc --gen-slurm \
		--n-samples $(TRAIN_SAMPLES) --cpus $(TRAIN_CPUS) --wall-time $(TRAIN_WALLTIME)
	@echo "Submit with: sbatch train_accelerator_$(TRAIN_SAMPLES).slurm"

slurm-train-container:
	$(PYTHON) -m prism3d.chemistry.train_hpc --gen-slurm \
		--container $(CONTAINER) \
		--n-samples $(TRAIN_SAMPLES) --cpus $(TRAIN_CPUS) --wall-time $(TRAIN_WALLTIME) \
		--outdir ./train_output
	@echo "Submit with: sbatch train_accelerator_$(TRAIN_SAMPLES).slurm"

# ── Clean ─────────────────────────────────────────────────

clean:
	rm -rf build/ dist/ *.egg-info __pycache__
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

clean-all: clean
	rm -rf test_output/ results/ orion_bar_data/ prepared/
	rm -f *.slurm *.pkl training_data_*.npz
