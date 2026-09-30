# Layer-wise truth probing in LLMs

2AMM20 Research Topics in Data Mining, TU/e, 2026 Q1. Understanding Large Language Models track, group 11.

**Research question.** How does the separability of true and false statements change across the layers of a large language model, and does this pattern differ between types of statements?

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

### Rebuilding

```
pip install -r requirements.txt
python scripts/build_dataset.py
```

### Source and licence

The raw CSVs in `data/raw/truth_is_universal` are the English datasets from the repository of Bürger, Hamprecht and Nadler, *Truth is Universal: Robust Detection of Lies in LLMs* (NeurIPS 2024), https://github.com/sciai-lab/Truth_is_Universal, released under the MIT licence (copy included). They collected most of the sets from earlier work, including Marks and Tegmark, *The Geometry of Truth* (2023), which should be cited alongside them.

Not included: the German translations, `common_claim_true_false`, `counterfact_true_false` and the real-world lie scenarios.

## Repository layout

```
data/
  raw/truth_is_universal/   source CSVs and licence
  statements.csv            combined dataset
  summary_by_type.csv       per-type counts, balance and length
  summary_by_topic.csv      statements per topic and type
docs/
  data_summary.md           readable data summary and caveats
scripts/
  build_dataset.py          builds everything in data/ and docs/ from the raw CSVs
```

Activations extracted from the models are large and are not stored in git (see `.gitignore`).
