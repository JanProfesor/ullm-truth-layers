# Layer-wise truth probing in LLMs

2AMM20 Research Topics in Data Mining, TU/e, 2026 Q1. Understanding Large Language Models track, group 11.

**Research question.** How does the separability of true and false statements change across the layers of a large language model, and does this pattern differ between types of statements?

## Setup

The project uses [uv](https://docs.astral.sh/uv/) for Python and dependencies. Install uv once (`winget install astral-sh.uv` on Windows, `curl -LsSf https://astral.sh/uv/install.sh | sh` on macOS and Linux), then from the repository root:

```
uv sync
```

This installs the Python version in `.python-version` if needed, creates `.venv` and installs the exact versions pinned in `uv.lock`. Run anything through `uv run`, for example:

```
uv run python scripts/build_dataset.py
```

Add a dependency with `uv add <package>` (for example `uv add torch transformers scikit-learn`) and commit the updated `pyproject.toml` and `uv.lock`.

## Data

`data/statements.csv` holds 17,049 labelled true/false statements in five statement types:

| Type | Example | Statements |
|---|---|---|
| affirmative | The city of Krasnodar is in Russia. | 3,150 |
| negated | The city of Krasnodar is not in Russia. | 3,148 |
| conjunction | It is the case both that the raccoon is a mammal and that the bison is a mammal. | 3,986 |
| disjunction | It is the case either that the slug is a mollusk or that it is a amphibian. | 2,805 |
| comparison | Fifty-one is larger than fifty-two. | 3,960 |

Affirmative, negated, conjunction and disjunction statements each cover six topics: `animal_class`, `cities`, `element_symb`, `facts`, `inventors` and `sp_en_trans`. Comparisons cover numbers only.

Columns:

| Column | Meaning |
|---|---|
| `id` | row id |
| `statement` | the sentence fed to the model |
| `label` | 1 = true, 0 = false |
| `type` | statement type (see above) |
| `topic` | knowledge topic |
| `source_file` | original CSV in `data/raw/truth_is_universal` |
| `group` | key for grouped train/test splits: statements that share it must stay on the same side |
| `n_words`, `n_chars` | length of the statement |

Counts, class balance, sentence lengths and the caveats to keep in mind are in [`docs/data_summary.md`](docs/data_summary.md).

### Rebuilding the data

```
uv run python scripts/build_dataset.py
```

### Source and licence

The raw CSVs in `data/raw/truth_is_universal` are the English datasets from the repository of Bürger, Hamprecht and Nadler, *Truth is Universal: Robust Detection of Lies in LLMs* (NeurIPS 2024), https://github.com/sciai-lab/Truth_is_Universal, released under the MIT licence (copy included). They collected most of the sets from earlier work, including Marks and Tegmark, *The Geometry of Truth* (2023), which should be cited alongside them.

Not included: the German translations, `common_claim_true_false`, `counterfact_true_false` and the real-world lie scenarios.

## Running the pipeline

The pipeline has three steps:

1. **Extract activations.** Run the language model on every statement and save its internal state (the hidden state of the last token) at every layer. This is the only step that needs a GPU for the real model.
2. **Train classifiers.** At every layer, train two simple classifiers (logistic regression and a linear SVM) to tell true from false statements, and measure their accuracy on statements they have not seen.
3. **Make PCA animations.** For each statement type, and for all types together, make a GIF that steps through the layers and shows how true and false statements move apart.

Steps 2 and 3 run on any laptop. The commands below are the same on macOS (Terminal) and Windows (PowerShell). Run them from the repository folder. Every script explains its options with `--help`.

### One-time setup

1. Install uv:
   - macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows: `winget install astral-sh.uv`
2. Open a new terminal, go to the repository folder and install everything:

```
uv sync
```

### Quick test on your own computer (about 5 minutes)

This runs the whole pipeline with a tiny model (SmolLM2-135M, downloaded automatically) on 1,000 statements. Use it to check that everything works before using the real model.

```
uv run python scripts/extract_activations.py --model HuggingFaceTB/SmolLM2-135M --out activations/smollm2-test --limit 1000
uv run python scripts/probe_layers.py --acts activations/smollm2-test --out results/smollm2-test
uv run python scripts/pca_gifs.py --acts activations/smollm2-test --out results/smollm2-test
```

The figures appear in `results/smollm2-test/`. The scripts pick the fastest hardware available:

- **Mac with Apple Silicon:** uses the built-in GPU (MPS).
- **Windows:** uses the CPU. This is fine for the small test model, because the standard PyTorch install on Windows has no GPU support.

Test results are not committed to git.

### Extracting activations of Llama 3 on Snellius

Llama 3 8B is too big for a laptop, so step 1 runs on the Snellius cluster. The model weights are already in `~/models/Meta-Llama-3-8B-Instruct` there. Log in with `ssh <user>@snellius.surf.nl`, which works from the macOS Terminal and from Windows PowerShell, then:

```
cd ullm-truth-layers
git pull
uv sync
mkdir -p logs
sbatch scripts/extract_activations.sbatch --limit 200 --out activations/test
```

This submits a small test job. Check it with `squeue -u $USER`: the job is listed while it waits or runs and disappears when it is done. Then read its log with `cat logs/extract_acts_<jobid>.out`.

If the log shows no errors, start the full run (all 17,049 statements, about 4.6 GB of output):

```
sbatch scripts/extract_activations.sbatch
```

If the GPU runs out of memory, add `--batch-size 32` to that command.

### Copying the activations to your computer

Run this on your own computer (macOS or Windows):

```
mkdir activations
scp -r <user>@snellius.surf.nl:ullm-truth-layers/activations/llama3-8b-instruct activations/
```

`mkdir` only complains if the folder already exists; that is harmless. The `activations/` folder is ignored by git: never commit activations.

### Running the analysis on the Llama activations

```
uv run python scripts/probe_layers.py --acts activations/llama3-8b-instruct --out results/llama3-8b-instruct
uv run python scripts/pca_gifs.py --acts activations/llama3-8b-instruct --out results/llama3-8b-instruct
```

Run `probe_layers.py` first, because the GIFs show its accuracy curves. `results/README.md` explains how to read the figures.

### How the classifiers are tested

- **Grouped 5-fold cross-validation.** Each classifier is trained on 80% of the statements and tested on the remaining 20%. This is repeated five times so every statement is tested once, and the reported accuracy is the average.
- **Related statements stay together.** Statements that share a `group` value never end up on both sides of a split. Examples are all statements about one city, a fact and its negation, or "51 is larger than 52" and "51 is smaller than 52". This stops a classifier from scoring well just by remembering a city or a number pair.
- **Reading the numbers.** 50% accuracy is chance. A layer where accuracy is high is a layer where the model represents truth in a form a simple classifier can read.

### Fixed design choices for the extraction

- **Input:** each statement is given as plain text, without a chat template.
- **Position:** we keep the hidden state of the last token of each statement at every layer, meaning the embedding output plus one per transformer block (33 for Llama 3 8B).
- **Layer 0 sanity check:** almost every statement ends in ".", so at layer 0 the last token is the same for nearly all statements. Accuracy there should be at chance level.

## Repository layout

```
data/
  raw/truth_is_universal/   source CSVs and licence
  statements.csv            combined dataset
  summary_by_type.csv       per-type counts, balance and length
  summary_by_topic.csv      statements per topic and type
docs/
  data_summary.md           readable data summary and caveats
results/
  README.md                 how to read each figure
  <model>/                  figures and CSVs per model (results/*-test/ is not committed)
scripts/
  build_dataset.py          builds everything in data/ and docs/ from the raw CSVs
  common.py                 shared helpers for the analysis scripts
  extract_activations.py    step 1: activations at every layer
  extract_activations.sbatch  step 1 as a Snellius job (1 A100 GPU)
  probe_layers.py           step 2: logistic regression and SVM accuracy per layer
  pca_gifs.py               step 3: PCA animation through the layers
pyproject.toml              project metadata and dependencies
uv.lock                     pinned dependency versions
.python-version             Python version used by uv
```

Activations extracted from the models are large and are not stored in git (see `.gitignore`).
