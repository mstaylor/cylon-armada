SHELL=/bin/bash

# Docker Hub username — override with: make image-pull DOCKER_USER=yourname
DOCKER_USER ?= mstaylor

SCRATCH     ?= /scratch/$(USER)
CPU_SIF     ?= $(SCRATCH)/cylon-armada/cylon-armada.sif
GPU_SIF     ?= $(SCRATCH)/cylon-armada/cylon-armada-gpu.sif
CPU_RESULTS ?= $(SCRATCH)/armada-results
GPU_RESULTS ?= $(SCRATCH)/armada-results-gpu
REDIS_HOST  ?= dev-cylon-redis1.aws-cylondata.com

.PHONY: image-pull image-pull-gpu smoke-cpu smoke-gpu cpu gpu q qq cancel clean

# ---------------------------------------------------------------------------
# Apptainer image management
# Pull from Docker Hub — run these from a Rivanna login node (not compute).
#
# APPTAINER_CACHEDIR and APPTAINER_TMPDIR are redirected to scratch so that
# layer decompression (~2-3x SIF size) does not fill the home directory quota.
# CPU SIF ~6 GB, GPU SIF ~20 GB; cache overhead can reach 50-70 GB total.
# ---------------------------------------------------------------------------

APPTAINER_CACHE ?= $(SCRATCH)/apptainer-cache
APPTAINER_TMP   ?= $(SCRATCH)/apptainer-tmp

image-pull:
	mkdir -p $(APPTAINER_CACHE) $(APPTAINER_TMP) $(dir $(CPU_SIF))
	module load apptainer && \
	APPTAINER_CACHEDIR=$(APPTAINER_CACHE) \
	APPTAINER_TMPDIR=$(APPTAINER_TMP) \
	apptainer pull $(CPU_SIF) docker://docker.io/$(DOCKER_USER)/cylon-armada-python:latest

image-pull-gpu:
	mkdir -p $(APPTAINER_CACHE) $(APPTAINER_TMP) $(dir $(GPU_SIF))
	module load apptainer && \
	APPTAINER_CACHEDIR=$(APPTAINER_CACHE) \
	APPTAINER_TMPDIR=$(APPTAINER_TMP) \
	apptainer pull $(GPU_SIF) docker://docker.io/$(DOCKER_USER)/cylon-armada-gpu:latest

image-cache-clean:
	rm -rf $(APPTAINER_CACHE) $(APPTAINER_TMP)
	@echo "Apptainer cache cleared from scratch"

# ---------------------------------------------------------------------------
# Smoke tests — 3 tasks, world_size=1, direct comparison with ECS smoke tests
# ---------------------------------------------------------------------------

smoke-cpu:
	mkdir -p logs
	python armada-experiment-setup.py --smoke \
		-d $(CPU_SIF) \
		-r1 $(REDIS_HOST) \
		-n 1 -t 1 -c 4 --time 00:30:00 \
		-m 16G \
		-l1 $(CPU_RESULTS) -l2 /output \
		-p2 standard

smoke-gpu:
	mkdir -p logs
	python armada-experiment-setup.py --smoke --gpu \
		-d $(GPU_SIF) \
		-r1 $(REDIS_HOST) \
		-n 1 -t 1 -c 4 --time 00:30:00 \
		-m 16G \
		-l1 $(GPU_RESULTS) -l2 /output \
		-p2 bii-gpu

# ---------------------------------------------------------------------------
# Full experiment sweeps (generate + submit)
# ---------------------------------------------------------------------------

cpu:
	mkdir -p logs
	python armada-experiment-setup.py \
		-d $(CPU_SIF) \
		-r1 $(REDIS_HOST) \
		-n 1 -t 1 -c 10 \
		--tasks 4 8 16 --thresholds 0.8 --dimensions 256 \
		--backends NUMPY --runs 3 --scenario hydrology \
		-l1 $(CPU_RESULTS) -l2 /output \
		-p2 standard --dry-run

gpu:
	mkdir -p logs
	python armada-experiment-setup.py --gpu \
		-d $(GPU_SIF) \
		-r1 $(REDIS_HOST) \
		-n 1 -t 1 -c 10 \
		--tasks 4 8 16 --thresholds 0.8 --dimensions 256 \
		--backends GCYLON --runs 3 --scenario hydrology \
		-l1 $(GPU_RESULTS) -l2 /output \
		-p2 bii-gpu --dry-run

# ---------------------------------------------------------------------------
# SLURM management
# ---------------------------------------------------------------------------

q:
	squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me

qq:
	watch squeue --format=\"%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R\" --me

cancel:
	scancel -u $(USER)

clean:
	rm -f *.log *.err script-*.slurm submit.log