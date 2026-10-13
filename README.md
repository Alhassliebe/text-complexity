# Text Complexity and Authorial Style in Russian Prose

Code for my bachelor's thesis at HSE University (2025), *Integral and discrete characteristics of text complexity*.

The question: can a set of interpretable complexity metrics (syntax, vocabulary, morphology, style) tell authors apart? I compute 18 metrics on dependency-parsed Russian texts, aggregate them per text and check whether texts cluster by author.

## Data

201 Russian literary texts by 33 authors, from Pushkin and Dostoevsky to Akunin and Lukyanenko, in CoNLL-U format (Universal Dependencies). The corpus is not included in the repository. Put the `Authors_conllu` folder into `data/`.

## Method

**Micro level** (`micro_level_analysis.py`). Each text is split into segments of 10 and 100 sentences, and 18 metrics are computed for every segment:

| Group | Metrics |
|---|---|
| Syntactic | average sentence length, dependency tree depth, average dependency distance, subordinate and coordinate constructions per sentence |
| Lexical | type-token ratio, lexical density, hapax legomena share, average word length, syllables per word |
| Morphological | morphological complexity, passive voice share, personal pronoun share |
| Stylistic | parts of speech per sentence, modifiers per noun, punctuation and discourse markers per sentence |
| Integral | Flesch reading ease (Russian adaptation) |

Formulas and interpretation are in [`description_of_metrics.md`](description_of_metrics.md).

**Macro level** (`macro_level_analysis.py`). Metrics are aggregated per text, normalised, reduced with t-SNE and UMAP, clustered with K-means and hierarchical clustering, and evaluated against author labels (homogeneity, completeness, adjusted Rand index).

## Results

| Segments | Method | Homogeneity | Completeness | ARI |
|---|---|---|---|---|
| 10 sentences | K-means | 0.102 | 0.693 | 0.019 |
| 10 sentences | Hierarchical | 0.111 | 0.673 | 0.023 |
| 100 sentences | K-means | 0.103 | 0.720 | 0.020 |
| 100 sentences | Hierarchical | 0.113 | 0.749 | 0.023 |

Complexity metrics alone give a weak but consistent grouping by author. Hierarchical clustering does slightly better than K-means, and longer segments (100 sentences) work better on every measure. Full report: [`report.md`](report.md) (in Russian).

## How to run

```bash
pip install -r requirements.txt
python micro_level_analysis.py   # metrics per segment, saved to output/
python macro_level_analysis.py   # aggregation, clustering and report
```

In `main()` of `micro_level_analysis.py` you can switch between processing the whole corpus and a single file for a quick demo.

## Tools

Python, conllu, Udapi, Natasha, razdel, pandas, NumPy, SciPy, scikit-learn, UMAP, Matplotlib, seaborn.
