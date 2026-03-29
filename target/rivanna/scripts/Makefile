SHELL=/bin/bash

.PHONY: load clean q qq cancel

# ---------------------------------------------------------------------------
# Apptainer image
# ---------------------------------------------------------------------------

image-pull:
	module load apptainer && \
	apptainer pull cylon-armada.sif docker://docker.io/$(DOCKER_USER)/cylon-armada-python:latest

image-pull-gpu:
	module load apptainer && \
	apptainer pull cylon-armada-gpu.sif docker://docker.io/$(DOCKER_USER)/cylon-armada-gpu:latest

# ---------------------------------------------------------------------------
# Experiment setup (generates SLURM scripts)
# ---------------------------------------------------------------------------

cpu:
	python armada-experiment-setup.py \
		-d cylon-armada.sif \
		-r1 localhost -t 1 -n 1 -c 10 \
		--tasks 4 8 16 --thresholds 0.8 --dimensions 256 \
		--runs 3 --scenario hydrology \
		-l1 /scratch/$(USER)/armada-results -l2 /output \
		-p2 standard --dry-run

gpu:
	python armada-experiment-setup.py \
		-d cylon-armada-gpu.sif \
		-r1 localhost -t 1 -n 1 -c 10 \
		--tasks 4 8 16 --thresholds 0.8 --dimensions 256 \
		--runs 3 --scenario hydrology --gpu \
		-l1 /scratch/$(USER)/armada-results-gpu -l2 /output \
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
	rm -f *.log *.err script-*.slurm