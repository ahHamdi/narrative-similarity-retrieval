# Narrative Similarity

![Python](https://img.shields.io/badge/Python-3.9%2B%20(tested%203.12)-blue)


## Track A – Embedding-Based Similarity Baseline

This script runs an embedding-based similarity baseline for **Track A** using
[SentenceTransformers](https://www.sbert.net/).

Given an `anchor_text` and two candidate texts (`text_a`, `text_b`), the script
predicts which candidate is closer to the anchor using cosine similarity of
sentence embeddings.

The output is written as **JSONL** and matches the expected evaluation format.

### Evaluation

```bash
python run_track_a.py \
  --model your model
```

To add a model, modify `MODEL_CHOICES` in `run_track_a.py`.

## Track B – Embedding Vector Generation Baseline

This script runs an embedding-based baseline for **Track B** using
[SentenceTransformers](https://www.sbert.net/).

Given an `anchor_text`, the script produces a vector representation. These representations should have a cosine similarity that aligns with the underlying stories’ narrative similarities.

The output is a set of embeddings in a **JSONL** (a list of floats in the embeddings property) or numpy serialized file **npy**.
This matches the expected evaluation format.

### Evaluation

```bash
python run_track_b.py \
  --model your_model
```

To add a model, modify `MODEL_CHOICES` in `run_track_b.py`.

# Information retreival

![Python](https://img.shields.io/badge/Python-3.9%2B%20(tested%203.12)-blue)

This script runs an embedding-based baseline to retreive relevant documents giveing a query.

Given a `query` the script predicts which candidate documents are relevant. It then evaluates results compared to the ground truth and returns IR metrics

To add a model, modify `models` in `information_retreival.py`.

To run experiments on Tell me again! You have to download the dataset from the original paper of (Hatzel et al. 2024) and run :

```bash
python evaluate_tma.py 
```

## License
This dataset is derived from Wikipedia and is released under the 
[Creative Commons Attribution-ShareAlike 4.0 International License (CC-BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).
