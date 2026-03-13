"""
eval_ladder_re.py — Evaluate LADDER Relation Extraction on test data.

Usage (called by run.py):
    python eval_ladder_re.py [limit]

Produces: outputs/ladder_re_none_results.json
Output format matches eval_ctinexus.py:
    [{file, text, ontology, extracted_entities, extracted_triplets}, ...]
"""

import os
import sys
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import AutoTokenizer

from networks import RelationClassificationBERT, RelationClassificationRoBERTa

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent.resolve()
_ROOT = _HERE.parent.parent.parent  # watson-eval root
OUTPUT_DIR = os.getenv("OUTPUT_DIR", str(_ROOT / "outputs"))
BASELINE_NAME = "ladder_re"

# ── Model configuration (must match train_supervised.py) ──────────────────────
MODEL = "bert-base-uncased"
NUM_LABELS = 11
MAX_LENGTH = 512
BATCH_SIZE = 8
CHECKPOINT = str(_HERE / "150_best_1.pt")

RELATION_LABELS = [
    "N/A", "isA", "targets", "uses", "hasAuthor",
    "has", "variantOf", "hasAlias", "indicates",
    "discoveredIn", "exploits",
]

DATASET = "150"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_entity_pair(sentence: str):
    """Extract the two marked entities (e1, e2) from the sentence."""
    # Sentences contain markers like: ... <e1>EntityName</e1> ... <e2>EntityName</e2> ...
    e1_match = re.search(r"<e1>(.+?)</e1>", sentence)
    e2_match = re.search(r"<e2>(.+?)</e2>", sentence)
    e1_text = e1_match.group(1).strip() if e1_match else ""
    e2_text = e2_match.group(1).strip() if e2_match else ""
    return e1_text, e2_text


def run_evaluation(limit=None):
    """Load trained model, run inference on test split, save standardised results."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load test data ─────────────────────────────────────────────────────────
    data_dir = _HERE / "data" / DATASET
    sentence_test = json.load(open(data_dir / "test_sentence.json", "r"))
    label_test = json.load(open(data_dir / "test_label_id.json", "r"))

    if limit:
        sentence_test = sentence_test[:limit]
        label_test = label_test[:limit]

    print(f"Test set: {len(sentence_test)} samples")

    # ── Tokenise ───────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Determine entity marker token IDs
    if MODEL in ["bert-base-uncased", "bert-large-uncased"]:
        e1_id, e2_id = 2487, 2475
    elif MODEL in ["roberta-base", "roberta-large"]:
        e1_id, e2_id = 134, 176
    elif MODEL in ["xlm-roberta-base", "xlm-roberta-large"]:
        e1_id, e2_id = 418, 304
    else:
        raise ValueError(f"Unknown model: {MODEL}")

    input_ids_list, attention_masks_list, labels_list = [], [], []
    e1_pos_list, e2_pos_list = [], []

    for i in range(len(sentence_test)):
        encoded = tokenizer(
            sentence_test[i],
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids_list.append(encoded["input_ids"])
        attention_masks_list.append(encoded["attention_mask"])
        labels_list.append(label_test[i])

        try:
            e1_idx = (encoded["input_ids"] == e1_id).nonzero(as_tuple=True)[1][0].item()
        except Exception:
            e1_idx = 0
        try:
            e2_idx = (encoded["input_ids"] == e2_id).nonzero(as_tuple=True)[1][0].item()
        except Exception:
            e2_idx = 0

        e1_pos_list.append(e1_idx)
        e2_pos_list.append(e2_idx)

    input_ids = torch.cat(input_ids_list, dim=0).to(device)
    attention_masks = torch.cat(attention_masks_list, dim=0).to(device)
    labels = torch.tensor(labels_list, device=device)
    e1_pos = torch.tensor(e1_pos_list, device=device)
    e2_pos = torch.tensor(e2_pos_list, device=device)
    w = torch.ones(len(e1_pos_list), device=device)

    test_dataset = TensorDataset(input_ids, attention_masks, labels, e1_pos, e2_pos, w)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=BATCH_SIZE,
    )

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"Loading model checkpoint: {CHECKPOINT}")
    if "roberta" in MODEL:
        model = RelationClassificationRoBERTa.from_pretrained(
            MODEL, num_labels=NUM_LABELS,
            output_attentions=False, output_hidden_states=False,
        )
    else:
        model = RelationClassificationBERT.from_pretrained(
            MODEL, num_labels=NUM_LABELS,
            output_attentions=False, output_hidden_states=False,
        )

    model = torch.nn.DataParallel(model)
    model.to(device)

    # Load checkpoint weights
    if os.path.exists(CHECKPOINT):
        state_dict = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded successfully")
    else:
        print(f"WARNING: Checkpoint not found at {CHECKPOINT}, using base model")

    model.eval()

    # ── Inference ──────────────────────────────────────────────────────────────
    all_predictions = []
    all_labels = []

    print("Running inference …")
    with torch.no_grad():
        for batch in test_dataloader:
            b_input_ids, b_masks, b_labels, b_e1, b_e2, b_w = batch
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_masks,
                labels=b_labels,
                e1_pos=b_e1,
                e2_pos=b_e2,
                w=b_w,
            )
            logits = outputs[1]
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_predictions.extend(preds.tolist())
            all_labels.extend(b_labels.cpu().numpy().tolist())

    # ── Build standardised results ─────────────────────────────────────────────
    seen_entities = set()
    extracted_entities = []
    extracted_triplets = []

    for i, (sent, pred_id) in enumerate(zip(sentence_test, all_predictions)):
        pred_label = RELATION_LABELS[pred_id] if pred_id < len(RELATION_LABELS) else "unknown"

        # Skip N/A predictions
        if pred_label == "N/A":
            continue

        e1_text, e2_text = extract_entity_pair(sent)
        if not e1_text or not e2_text:
            continue

        # Collect entities
        for ent_name in [e1_text, e2_text]:
            if ent_name not in seen_entities:
                extracted_entities.append({"name": ent_name, "class": "unknown"})
                seen_entities.add(ent_name)

        # Collect triplet
        extracted_triplets.append({
            "subject": e1_text,
            "relation": pred_label,
            "relation_class": pred_label,
            "object": e2_text,
        })

    # Compute simple metrics
    correct = sum(1 for p, l in zip(all_predictions, all_labels) if p == l)
    accuracy = correct / len(all_predictions) if all_predictions else 0

    results = [{
        "file": f"ladder_re_test_{DATASET}",
        "text": f"Relation Extraction test on {DATASET} dataset ({len(sentence_test)} samples, accuracy={accuracy:.4f})",
        "ontology": "none",
        "extracted_entities": extracted_entities,
        "extracted_triplets": extracted_triplets,
        "raw_output": {
            "model": MODEL,
            "dataset": DATASET,
            "total_samples": len(sentence_test),
            "accuracy": round(accuracy, 4),
            "relation_distribution": {
                RELATION_LABELS[i]: all_predictions.count(i)
                for i in range(NUM_LABELS)
            },
        },
    }]

    # Save
    output_file = os.path.join(OUTPUT_DIR, f"{BASELINE_NAME}_none_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults: {len(extracted_triplets)} triplets, {len(extracted_entities)} unique entities")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Evaluation complete. Results saved to {output_file}")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    limit = limit if limit > 0 else None
    run_evaluation(limit=limit)
