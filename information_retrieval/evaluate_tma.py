#Evaluation of pretrained and PLOTS-trained models on the Tell Me Again! external benchmark.
import os
import json
import zipfile
import torch
from collections import defaultdict

from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator

ZIP_PATH  = "tma/tell_me_again_v1.zip"
MIN_SUMMARIES = 2
MIN_TEXT_LEN  = 50

print(f"Loading stories from {ZIP_PATH}...")

story_summaries = []

with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    json_files = [f for f in zf.namelist()
                  if f.startswith("summaries/") and f.endswith(".json")]
    print(f"Found {len(json_files)} story JSON files.")

    for jf in json_files:
        with zf.open(jf) as f:
            story = json.load(f)

        translated = story.get("en_translated_summaries", {})
        texts = []
        for lang, content in translated.items():
            text = content.get("text", "").strip()
            if len(text) >= MIN_TEXT_LEN:
                texts.append(text)

        native_en = story.get("summaries", {}).get("en", "").strip()
        if len(native_en) >= MIN_TEXT_LEN:
            texts.append(native_en)

        if len(texts) >= MIN_SUMMARIES:
            story_summaries.append(texts)

print(f"Stories with >= {MIN_SUMMARIES} summaries: {len(story_summaries)}")

queries       = {}
corpus        = {}
relevant_docs = {}
qid    = 0
doc_id = 0

for texts in story_summaries:
    doc_ids_for_story = []
    for text in texts:
        corpus[str(doc_id)] = text
        doc_ids_for_story.append(str(doc_id))
        doc_id += 1
    queries[str(qid)]       = texts[0]
    relevant_docs[str(qid)] = set(doc_ids_for_story[1:])
    qid += 1

avg_rel = sum(len(v) for v in relevant_docs.values()) / len(relevant_docs)
print(f"\nIR Benchmark:")
print(f"  Queries  : {len(queries)}")
print(f"  Corpus   : {len(corpus)}")
print(f"  Avg relevant docs/query: {avg_rel:.2f}\n")

models = [
    # Zero-shot baselines
    ("sentence-transformers/all-MiniLM-L6-v2", "minilm-6 (zero-shot)",      False),
    ("BAAI/bge-large-en-v1.5",                 "bge-large-en (zero-shot)",   False),
    ("nomic-ai/nomic-embed-text-v1",           "nomic (zero-shot)",          True),
    ("sentence-transformers/all-mpnet-base-v2","mpnet-base (zero-shot)",     False),

    # PLOTS-finetuned: minilm-6
    ("ahmedHamdi/IR-pt-en-miniLM-L6",  "minilm-6 EN-PT (full text)",  False),
    ("ahmedHamdi/IR-es-en-miniLM-L6",  "minilm-6 EN-SP (full text)",  False),
    ("ahmedHamdi/IR-fr-en-miniLM-L6",  "minilm-6 EN-FR (full text)",  False),
    ("ahmedHamdi/IR-all-en-miniLM-L6", "minilm-6 EN-ALL (full text)", False),

    # PLOTS-finetuned: instructor-xl
    ("ahmedHamdi/ir-pt-en-instructor-xl",  "instructor-xl EN-PT (full text)",  False),
    ("ahmedHamdi/ir-es-en-instructor-xl",  "instructor-xl EN-SP (full text)",  False),
    ("ahmedHamdi/ir-fr-en-instructor-xl",  "instructor-xl EN-FR (full text)",  False),
    ("ahmedHamdi/ir-all-en-instructor-xl", "instructor-xl EN-ALL (full text)", False),

    # PLOTS-finetuned: gemma
    ("ahmedHamdi/IR-pt-en-gemma",  "gemma EN-PT (full text)",  False),
    ("ahmedHamdi/IR-es-en-gemma",  "gemma EN-SP (full text)",  False),
    ("ahmedHamdi/IR-fr-en-gemma",  "gemma EN-FR (full text)",  False),
    ("ahmedHamdi/IR-all-en-gemma", "gemma EN-ALL (full text)", False),
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

results = []

for model_name, label, trust_remote in models:
    print(f"Evaluating: {label}", flush=True)
    try:
        # Load from local cache only
        local_path = snapshot_download(
            repo_id=model_name,
            cache_dir=HF_CACHE,
            local_files_only=True,
        )
        model = SentenceTransformer(
            local_path,
            device=device,
            trust_remote_code=trust_remote,
        )

        evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name="tell-me-again",
            mrr_at_k=[10],
            ndcg_at_k=[10],
            show_progress_bar=True,
        )

        scores   = evaluator.compute_metrices(model)
        cos_sim  = scores.get("cos_sim", {})
        ndcg     = cos_sim.get("ndcg@k", {}).get(10, None)
        mrr      = cos_sim.get("mrr@k",  {}).get(10, None)

        results.append({
            "model":    label,
            "nDCG@10": round(float(ndcg), 4) if ndcg is not None else "N/A",
            "MRR@10":  round(float(mrr),  4) if mrr  is not None else "N/A",
        })
        print(f"  nDCG@10: {ndcg:.4f}  |  MRR@10: {mrr:.4f}\n", flush=True)

    except Exception as e:
        print(f"  ERROR: {e}\n", flush=True)
        results.append({"model": label, "nDCG@10": "ERROR", "MRR@10": "ERROR"})

    finally:
        # Free GPU memory between models
        try:
            del model
        except:
            pass
        torch.cuda.empty_cache()

print("\n" + "=" * 65)
print("RESULTS ON TELL ME AGAIN!")
print("=" * 65)
print(f"{'Model':<45} {'nDCG@10':>10} {'MRR@10':>10}")
print("-" * 65)
for r in results:
    print(f"{r['model']:<45} {str(r['nDCG@10']):>10} {str(r['MRR@10']):>10}")
print("=" * 65)
