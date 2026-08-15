SHELL=/bin/bash

# Python: Intel Distribution for Python on Rivanna (stdlib only needed).
# Load before running any make target:
#   module load intel-compilers/2023.1.0 impi/2021.9.0 python/3.9.16
# All complex dependencies (cylon_dev, pycylon, cudf) live inside the Apptainer SIF.
PYTHON ?= python

# Docker Hub username — override with: make image-pull DOCKER_USER=yourname
DOCKER_USER ?= mstaylor

SCRATCH      ?= /scratch/$(USER)
GPU_SIF      ?= $(SCRATCH)/cylon-armada/cylon-armada-gpu.sif
UCCUCX_SIF   ?= $(SCRATCH)/cylon-armada/cylon-armada-uccucx.sif
CPU_RESULTS  ?= $(SCRATCH)/armada-results
GPU_RESULTS  ?= $(SCRATCH)/armada-results-gpu
EXPB_RESULTS ?= $(SCRATCH)/expb-results
REDIS_HOST   ?= dev-cylon-redis1.aws-cylondata.com
# AWS credentials: run 'aws configure' on Rivanna login node once.
# Stored in ~/.aws/credentials (home dir, NFS-mounted on all compute nodes).

.PHONY: image-pull-gpu image-pull-uccucx smoke-cpu smoke-gpu expb-smoke cpu cpu-all gpu gpu-all q qq cancel clean

# ---------------------------------------------------------------------------
# Apptainer image management
# Pull from Docker Hub — run these from a Rivanna login node (not compute).
#
# APPTAINER_CACHEDIR and APPTAINER_TMPDIR are redirected to scratch so that
# layer decompression (~2-3x SIF size) does not fill the home directory quota.
# GPU SIF ~20 GB, UCC/UCX SIF ~6-7 GB; cache overhead can reach 50-70 GB total.
#
# cylon-armada-uccucx-python serves BOTH the agentic pipeline (cpu/cpu-all/
# smoke-cpu) and Experiment B collectives (expb-smoke) on Rivanna — it's a
# strict superset of the FMI-only image (same shared scripts, same
# armada_ecs_runner.py, same FMI support, plus UCX/UCC), and Rivanna has none
# of Lambda's constraints against running UCC/UCX. The FMI-only image
# (cylon-armada-fmi-python) is Lambda-specific; there's no reason to pull it
# here too.
# ---------------------------------------------------------------------------

APPTAINER_CACHE ?= $(SCRATCH)/apptainer-cache
APPTAINER_TMP   ?= $(SCRATCH)/apptainer-tmp

image-pull-gpu:
	mkdir -p $(APPTAINER_CACHE) $(APPTAINER_TMP) $(dir $(GPU_SIF))
	rm -f $(GPU_SIF)
	module load apptainer && \
	APPTAINER_CACHEDIR=$(APPTAINER_CACHE) \
	APPTAINER_TMPDIR=$(APPTAINER_TMP) \
	apptainer pull $(GPU_SIF) docker://docker.io/$(DOCKER_USER)/cylon-armada-gpu:latest

image-pull-uccucx:
	mkdir -p $(APPTAINER_CACHE) $(APPTAINER_TMP) $(dir $(UCCUCX_SIF))
	rm -f $(UCCUCX_SIF)
	module load apptainer && \
	APPTAINER_CACHEDIR=$(APPTAINER_CACHE) \
	APPTAINER_TMPDIR=$(APPTAINER_TMP) \
	apptainer pull $(UCCUCX_SIF) docker://docker.io/$(DOCKER_USER)/cylon-armada-uccucx-python:latest

image-cache-clean:
	rm -rf $(APPTAINER_CACHE) $(APPTAINER_TMP)
	@echo "Apptainer cache cleared from scratch"

# ---------------------------------------------------------------------------
# Smoke tests — 3 tasks, world_size=1, direct comparison with ECS smoke tests
# ---------------------------------------------------------------------------

smoke-cpu:
	mkdir -p logs
	$(PYTHON) armada-experiment-setup.py --smoke \
		-d $(UCCUCX_SIF) \
		-r1 $(REDIS_HOST) \
		-n 1 -t 1 -c 4 --time 00:30:00 \
		-m 16G \
		-l1 $(CPU_RESULTS) -l2 /output \
		-p2 standard

smoke-gpu:
	mkdir -p logs
	$(PYTHON) armada-experiment-setup.py --smoke --gpu \
		-d $(GPU_SIF) \
		-r1 $(REDIS_HOST) \
		-n 1 -t 1 -c 4 --time 00:30:00 \
		-m 16G \
		-l1 $(GPU_RESULTS) -l2 /output \
		-p2 bii-gpu

# Experiment B collectives — multi-rank (--ntasks=4) UCC connectivity smoke
# test. Real distributed ranks (not threads), self-assigned via redis-OOB.
EXPB_WORLD_SIZE ?= 4

expb-smoke:
	mkdir -p logs $(EXPB_RESULTS)
	sbatch --ntasks=$(EXPB_WORLD_SIZE) \
		--export=ALL,SIF_IMAGE=$(UCCUCX_SIF),REDIS_HOST=$(REDIS_HOST),RESULTS_DIR_HOST=$(EXPB_RESULTS) \
		run_exp_b_collectives.slurm

# ---------------------------------------------------------------------------
# Full experiment sweeps (generate + submit)
# Run once per scenario: make cpu SCENARIO=hydrology
#                        make cpu SCENARIO=epidemiology  etc.
# Or submit all at once: make cpu-all / make gpu-all
# ---------------------------------------------------------------------------

SCENARIO ?= hydrology

cpu:
	mkdir -p logs
	$(PYTHON) armada-experiment-setup.py \
		-d $(UCCUCX_SIF) \
		-r1 $(REDIS_HOST) \
		-n 1 -t 1 -c 10 \
		--tasks 8 16 --thresholds 0.70 0.80 0.90 \
		--dimensions 256 1024 \
		--backends NUMPY --runs 4 --scenario $(SCENARIO) \
		-l1 $(CPU_RESULTS) -l2 /output \
		-p2 standard

cpu-all:
	for scenario in hydrology epidemiology seismology mixed_scientific; do \
		mkdir -p logs && \
		$(PYTHON) armada-experiment-setup.py \
			-d $(UCCUCX_SIF) \
			-r1 $(REDIS_HOST) \
			-n 1 -t 1 -c 10 \
			--tasks 8 16 --thresholds 0.70 0.80 0.90 \
			--dimensions 256 1024 \
			--backends NUMPY --runs 4 --scenario $$scenario \
			-l1 $(CPU_RESULTS) -l2 /output \
			-p2 standard; \
	done

gpu:
	mkdir -p logs
	$(PYTHON) armada-experiment-setup.py --gpu \
		-d $(GPU_SIF) \
		-r1 $(REDIS_HOST) \
		-n 1 -t 1 -c 10 \
		--tasks 8 16 --thresholds 0.70 0.80 0.90 \
		--dimensions 256 1024 \
		--backends GCYLON --runs 4 --scenario $(SCENARIO) \
		-l1 $(GPU_RESULTS) -l2 /output \
		-p2 bii-gpu

gpu-all:
	for scenario in hydrology epidemiology seismology mixed_scientific; do \
		mkdir -p logs && \
		$(PYTHON) armada-experiment-setup.py --gpu \
			-d $(GPU_SIF) \
			-r1 $(REDIS_HOST) \
			-n 1 -t 1 -c 10 \
			--tasks 8 16 --thresholds 0.70 0.80 0.90 \
			--dimensions 256 1024 \
			--backends GCYLON --runs 4 --scenario $$scenario \
			-l1 $(GPU_RESULTS) -l2 /output \
			-p2 bii-gpu; \
	done

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
	rm -f *.log *.err script-*.slurm env-*.env submit.log
	rm -f logs/*.log logs/*.err