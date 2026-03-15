# Word2Vec Pure NumPy implementation

Skip-Gram and CBOW with negative sampling, implemented in pure NumPy.

## About

This is the repo for submission to three internship project applications: "Hallucination Detection", "Learning to Reason with Small Models", "Test Time Reinforcement Learning" and that is why I decided to implement two variants of Word2Vec: CBOW and Skip-Gram with negative sampling and frequent words subsampling.

Both variants are available to run via `word2vec.py`. The repository also contains `output_CBOW.txt` and `output_SKIP-GRAM.txt` with experiment results and similarity checks.

## Clone

```bash
git clone https://github.com/Whitley7/word2vec-numpy.git
cd word2vec-numpy
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy
```

## Download dataset

```bash
curl -O http://mattmahoney.net/dc/text8.zip
unzip text8.zip
```

## Run

```bash
python word2vec.py text8
```

## Implementation notes

**Dataset.** Text8 is 100 million characters from a Wikipedia dump, already cleaned and lowercased. That's why preprocessing only includes splitting into tokens — no further cleaning is needed.

**Hyperparameters.** All values follow the original paper and C implementation:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `embed_dim` | 100 | Standard range is 100–300. 100 is enough for text8, larger would overfit |
| `window_size` | 5 | Mikolov's default. Captures enough context without being too broad |
| `num_neg` | 5 | Paper recommends 5–20. 5 is sufficient for a large dataset |
| `lr0` | 0.025 | Exact value from the original C code |
| `min_lr` | 1e-4 | Floor so learning rate never hits zero |
| `min_count` | 5 | Words appearing <5 times don't have enough context for meaningful embeddings |
| `t` | 1e-5 | Subsampling threshold from the paper. Aggressively drops "the", "a", "of", etc. |
| `epochs` | 1 (SG) / 3 (CBOW) | Skip-gram gets multiple updates per position, so 1 epoch suffices. CBOW needs more since it gets one update per position |

**RuntimeWarnings.** The CBOW model may produce warnings during evaluation after few epochs. This happens because some embedding rows are still near-zero early in training, causing overflow in dot products. These are handled in code (`nan_to_num`) and do not affect results. More epochs eliminate them.

## References

- Mikolov et al., *Efficient Estimation of Word Representations in Vector Space* (2013)
- Mikolov et al., *Distributed Representations of Words and Phrases and their Compositionality* (2013)
- Rong, *word2vec Parameter Learning Explained* (2014) - main reference for understanding the math
