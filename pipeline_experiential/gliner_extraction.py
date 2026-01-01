import json
import re
import random
import sys
from pathlib import Path
from collections import defaultdict
from gliner import GLiNER
import pandas as pd

# Add project root to path for config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CORPUS_PATH, RESULTS_DIR

# config

MODEL_NAME = "urchade/gliner_small-v2.1"
RANDOM_SEED = 42

GLOBAL_THRESHOLD = 0.45

LABEL_THRESHOLDS = {
    "BODY": 0.50,
    "MEMORY": 0.60,
    "PLACE": 0.65,
    "PERSON": 0.65
}

LABELS = {
    "BODY": (
    "bodily references, encompassing both affective and sensory dimensions.."
    ),
    "MEMORY": (
        "references to temporal or mnemonic dimensions, including past events, recollections, or nostalgic framing."
    ),
    "PLACE": (
        "spatial locations or situational settings where listening occurs "
        "(e.g., home, car, bedroom, college, venue)."
    ),
    "PERSON": (
        "people or social roles involved in the listening experience "
        "(e.g., singer, listeners, fans, artist)."
    )
}

# ----------------------------
# Hard blockers + lexical gates
# ----------------------------

PRONOUNS = {
    "i", "me", "my", "myself",
    "you", "your", "yourself",
    "he", "him", "his",
    "she", "her", "hers",
    "they", "them", "their", "theirs",
    "we", "us", "our", "ours", "ourselves"
}

INSTRUMENT_FALSE_BODY = {
    "violin", "violins", "viola", "violas", "cello", "cellos",
    "guitar", "guitars", "drum", "drums", "piano", "pianos",
    "bass", "trumpet", "trumpets", "sax", "saxophone", "saxophones",
    "flute", "flutes", "clarinet", "clarinets"
}

BODY_LEXICAL_ANCHORS = {
    "body", "skin", "nerve", "nerves", "chest", "heart",
    "lungs", "breath", "breathing", "pulse", "heartbeat",
    "blood", "veins", "chills", "mouth","ear","nerves", "body of work", "tears", "hands", "freeze", "warmth","goosebumbs", "hair", "eyes"
}

# ----------------------------
# Paths (using config.py)
# ----------------------------

INPUT_PATH = CORPUS_PATH

RESULTS_TABLES_DIR = RESULTS_DIR / "tables"
RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_RESULTS = RESULTS_TABLES_DIR / "gliner_stratified_results.json"
OUTPUT_STATS = RESULTS_TABLES_DIR / "gliner_stratified_stats.json"
OUTPUT_TABLE = RESULTS_TABLES_DIR / "gliner_ci_tablenew.csv"
OUTPUT_SPANS_CSV = RESULTS_TABLES_DIR / "gliner_predicted_spans.csv"

R_REVIEWS_PER_SOURCE = 200
K_SENTENCES_PER_REVIEW = 3
BOOTSTRAP_ITERATIONS = 1000

MIN_TOKENS = 450
MAX_TOKENS = 650

# exclude quoted song lyrics 

QUOTE_PAIRS = [('"', '"'), ('/', '/'), ('\\', '\\'), ('<', '>')]
MIN_QUOTED_CHARS = 15


def spans_in_quotes(text):
    spans = []
    for open_q, close_q in QUOTE_PAIRS:
        pattern = re.compile(
            rf"{re.escape(open_q)}(.*?){re.escape(close_q)}",
            re.DOTALL
        )
        for m in pattern.finditer(text):
            if len(m.group(1)) >= MIN_QUOTED_CHARS:
                spans.append((m.start(), m.end()))
    return spans


def remove_quoted_spans(text):
    spans = spans_in_quotes(text)
    if not spans:
        return text

    cleaned = []
    last = 0
    for start, end in sorted(spans):
        cleaned.append(text[last:start])
        last = end
    cleaned.append(text[last:])
    return " ".join("".join(cleaned).split())


# tokens and sentences

def within_token_range(text):
    return MIN_TOKENS <= len(text.split()) <= MAX_TOKENS


def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)


def select_sentences(sentences, k):
    candidates = [s for s in sentences if 10 < len(s) < 300]
    random.shuffle(candidates)
    return candidates[:k]


# le bootstrap confidence intervals

def bootstrap_ci(values, iters=1000, alpha=0.05):
    n = len(values)
    means = []

    for _ in range(iters):
        sample = [values[random.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[int((1 - alpha / 2) * iters)]
    return lo, hi


# hard-coded post filtering

def is_blocked_entity(ent):
    text = ent["text"].lower().strip()
    label = ent["label"]

    if label == "PERSON" and text in PRONOUNS:
        return True

    if label == "BODY":
        if text in INSTRUMENT_FALSE_BODY:
            return True
        if not any(tok in text for tok in BODY_LEXICAL_ANCHORS):
            return True

    return False


def main():
    """
    Main execution function for GLiNER-based entity extraction.
    """
    print("Starting GLiNER entity extraction pipeline...")
    
    # load model
    random.seed(RANDOM_SEED)
    model = GLiNER.from_pretrained(MODEL_NAME)

    # load and stratify corpus
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    by_source = defaultdict(list)

    for rid, item in enumerate(data):
        if not item.get("review") or not item.get("source"):
            continue

        text = remove_quoted_spans(item["review"])
        if not within_token_range(text):
            continue

        item["clean_review"] = text
        item["rid"] = rid
        by_source[item["source"]].append(item)

    sampled_reviews = []

    for source in ["Amazon", "Pitchfork"]:
        pool = by_source.get(source, [])
        random.shuffle(pool)
        sampled_reviews.extend(pool[:R_REVIEWS_PER_SOURCE])

    # sentence sampling
    review_sentences = []

    for item in sampled_reviews:
        sentences = split_sentences(item["clean_review"])
        selected = select_sentences(sentences, K_SENTENCES_PER_REVIEW)
        review_sentences.append({
            "source": item["source"],
            "rid": item["rid"],
            "sentences": selected
        })

    # inference + span collection
    results = []
    review_hits = []
    span_rows = []

    label_names = list(LABELS.keys())

    for review in review_sentences:
        review_level_hits = defaultdict(int)

        for sent in review["sentences"]:
            entities = model.predict_entities(
                sent,
                label_names,
                label_descriptions=LABELS,
                threshold=GLOBAL_THRESHOLD
            )

            for ent in entities:
                label = ent["label"]
                score = ent.get("score", 0.0)

                if score < LABEL_THRESHOLDS[label]:
                    continue

                if is_blocked_entity(ent):
                    continue

                review_level_hits[label] = 1
                span_rows.append({
                    "source": review["source"],
                    "review_id": review["rid"],
                    "label": label,
                    "span_text": ent["text"],
                    "sentence": sent,
                    "start": ent.get("start"),
                    "end": ent.get("end"),
                    "score": score
                })

            if entities:
                results.append({
                    "source": review["source"],
                    "sentence": sent,
                    "entities": entities
                })

        review_hits.append({
            "source": review["source"],
            "hits": review_level_hits
        })

    # compute rates and ci
    bootstrap_results = defaultdict(dict)

    for source in ["Amazon", "Pitchfork"]:
        reviews_src = [r for r in review_hits if r["source"] == source]

        for label in label_names:
            binary = [r["hits"].get(label, 0) for r in reviews_src]
            rate = sum(binary) / len(binary)
            lo, hi = bootstrap_ci(binary, BOOTSTRAP_ITERATIONS)

            bootstrap_results[source][label] = {
                "rate": rate,
                "ci_95": [lo, hi]
            }

    # save outputs
    with open(OUTPUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_STATS, "w", encoding="utf-8") as f:
        json.dump(bootstrap_results, f, indent=2, ensure_ascii=False)

    pd.DataFrame(span_rows).to_csv(OUTPUT_SPANS_CSV, index=False)

    rows = []
    for source, labels in bootstrap_results.items():
        for label, stats in labels.items():
            rows.append({
                "source": source,
                "category": label,
                "rate": round(stats["rate"], 3),
                "ci_lower": round(stats["ci_95"][0], 3),
                "ci_upper": round(stats["ci_95"][1], 3)
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_TABLE, index=False)

    print("=== Hit rates with 95% bootstrap CI ===")
    print(df)
    print(f"\nSpan CSV saved to {OUTPUT_SPANS_CSV}")
    print(f"CI table saved to {OUTPUT_TABLE}")


if __name__ == "__main__":
    main()
