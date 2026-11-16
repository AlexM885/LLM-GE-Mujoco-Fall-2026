# Agentic Islands on SLURM

## Project goal

- Run several evolutionary "islands" in parallel.
- Each island generates and evaluates neural-network candidates.
- Candidates are evaluated using ExquisiteNetV2 training.
- A global archive keeps non-dominated candidates:
  - maximize accuracy
  - minimize parameter count

## Main configuration

The main run settings are in `run_AgenticIslands.sh`:

- `GENERATIONS`: number of evolutionary generations.
- `EPOCHS`: training epochs per candidate.
- `POPULATION_SIZE`: population size for each island.
- `WORKERS`: number of Python worker threads used by the coordinator.
- `CANDIDATES_PER_CYCLE`: candidates generated per island cycle.
- `BATCH_SIZE`: training batch size.
- `INFERENCE_SUBMISSION`: whether the legacy LLM operators may submit inference requests.
- `CONDA_ENV`: Conda environment used on the compute node.

Current defaults:

```text
GENERATIONS=2
EPOCHS=2
POPULATION_SIZE=4
WORKERS=4
CANDIDATES_PER_CYCLE=1
BATCH_SIZE=216
INFERENCE_SUBMISSION=true
CONDA_ENV=llm_guided_env
```

The Python command-line equivalents are:

```text
--generations
--epochs
--population-size
--workers
--candidates-per-cycle
--batch-size
--inference-submission / --no-inference-submission
```

## Island configuration

The default island definitions are in `src/islands/config.py`:

- `accuracy`: accuracy-focused mutation.
- `efficiency`: parameter-efficient mutation.
- `stability`: conservative-stability mutation.
- `novelty`: architecture-novelty crossover.

The coordinator is implemented in:

- `src/islands/coordinator.py`
- `src/islands/worker.py`
- `src/islands/archive.py`
- `src/islands/migration.py`

Candidate and fitness data models are in:

- `src/islands/models.py`

## Important files

- `run_AgenticIslands.sh`
  - SLURM submission script and compute-node job script.
  - Requests one GPU, one node, four CPUs, and 16 GB per GPU.
  - Loads CUDA and Anaconda.
  - Activates the Conda environment.
  - Runs `run_AgenticIslands.py`.
- `run_AgenticIslands.py`
  - Builds island configurations.
  - Creates the LLM generator.
  - Evaluates candidates by running ExquisiteNetV2 training.
  - Writes the final archive to a JSON file.
- `src/islands/`
  - Agentic-island implementation.
- `sota/ExquisiteNetV2/`
  - Dataset preparation and model-training code.
- `templates/`
  - Templates used by the legacy LLM generation operators.

## Dataset setup

The dataset is not included in the repository. Download the CIFAR-10
Python archive from the official CIFAR-10 website and extract it under:

```text
sota/ExquisiteNetV2/cifar-10-batches-py
```

The directory should contain files such as:

```text
batches.meta
data_batch_1
data_batch_2
data_batch_3
data_batch_4
data_batch_5
test_batch
```

`split.py` requires Pillow. Install it in the Python environment used for
dataset preparation:

```bash
python -m pip install Pillow
```

Convert the CIFAR-10 binary batches into image folders:

```bash
cd sota/ExquisiteNetV2
python split.py
```

This creates:

```text
sota/ExquisiteNetV2/cifar10
```

## Environment setup

The SLURM script currently expects this Conda environment:

```text
llm_guided_env
```

Check available environments:

```bash
module load anaconda3
conda info --envs
```

If the environment has a different name, set it when submitting:

```bash
CONDA_ENV=my_environment ./run_AgenticIslands.sh /absolute/path/to/cifar10
```

The project requires Python 3.12 and dependencies listed in `pyproject.toml`.

## Submit one SLURM job

Run from the repository root:

```bash
./run_AgenticIslands.sh /absolute/path/to/cifar10
```

For this repository's prepared dataset:

```bash
./run_AgenticIslands.sh "$PWD/sota/ExquisiteNetV2/cifar10"
```

Set run parameters on the same command:

```bash
GENERATIONS=5 \
EPOCHS=10 \
POPULATION_SIZE=8 \
WORKERS=4 \
./run_AgenticIslands.sh "$PWD/sota/ExquisiteNetV2/cifar10"
```

Use the correct Conda environment if necessary:

```bash
CONDA_ENV=my_environment \
GENERATIONS=5 \
EPOCHS=10 \
POPULATION_SIZE=8 \
WORKERS=4 \
./run_AgenticIslands.sh "$PWD/sota/ExquisiteNetV2/cifar10"
```

## SLURM resources

The current script requests:

```text
job name:       agentic-islands
time limit:     08:00:00
GPU:            1
constraint:     A100-40GB|A100-80GB|H100|H200|V100-16GB|RTX6000|A40|L40S
memory:         16G per GPU
nodes:          1
tasks:          1
CPUs:           4
output:         slurm-agentic-islands-<jobid>.out
```

The GPU constraint must match the feature names supported by the cluster.
The current list follows the working SLURM configuration used by this
project.

## Monitor the job

After submission, note the job ID printed by `sbatch`:

```bash
squeue -u "$USER"
```

Follow the output:

```bash
tail -f slurm-agentic-islands-<jobid>.out
```

Cancel a job if needed:

```bash
scancel <jobid>
```

## Outputs

The run writes:

```text
agentic_islands_results_<jobid>.json
```

Generated island workspaces are stored under:

```text
islands/
```

The SLURM log is:

```text
slurm-agentic-islands-<jobid>.out
```

## Troubleshooting checklist

- `sbatch: Invalid feature specification`
  - Confirm the `#SBATCH -C` feature names with the cluster administrator or
    a working project script.
- `EnvironmentNameNotFound`
  - Run `conda info --envs` after loading Anaconda.
  - Submit with `CONDA_ENV=<actual-name>`.
- Dataset path does not exist
  - Pass the absolute path to the generated `cifar10` directory.
- `No module named PIL`
  - Install Pillow with `python -m pip install Pillow`.
- `python: can't open file`
  - Run the launcher from the repository or use an absolute dataset path.
  - The script derives `ROOT_DIR` from its own location.
