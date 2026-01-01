import re
import csv
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

# Add project root to path for config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CORPUS_PATH, RESULTS_DIR

# music lyric quotes within reviews

QUOTE_PAIRS = [
    ('"', '"'),
    ('/', '/'),
    ('\\', '\\'),
    ('<', '>'),
]

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


def is_inside_any(span_start, span_end, quote_spans):
    return any(
        span_start >= q_start and span_end <= q_end
        for q_start, q_end in quote_spans
    )

# rules

RULES = [
    {
        "rule_name": "BODY",
        "category": "BODY",
        "pattern": r"\b(body|chest|in my gut|physically|gives me chills|heart racing|blood pumping|ears|mouth|chest|limbs|gut|goosebumps|tears|crying|cry|throat|leg|arm|breath|hand|heart|teeth|mouth|nose|eyes|eye|foot|feet|blood|feeling|hearing|touch|touching|tasting|taste it|gut|tongue|lips|shiver|shivers|sweat|sweats|shivering)\b",
    },
    {
        "rule_name": "MEMORY",
        "category": "MEMORY",
        "pattern": r"\b(reminds me of|made me think of|brings back memories|takes me back to|I remember|childhood|years ago|i recall|souvenir|long time ago|nostalgic|nostalgia|remembering|memories of|not forgotten|not forget)\b",
    },
    {
        "rule_name": "PLACE",
        "category": "PLACE",
        "pattern": r"\b(in the car|at home|in my room|on the road|club|stadium|concert hall|bedroom|my car|my bed|chair|house|city|hotel|hospital|school|college|university|bar|restaurant|room)\b",
    },
    {
        "rule_name": "PERSON",
        "category": "PERSON",
        "pattern": r"\b("
                   r"my friend|my friends|"
                   r"my partner|my partners|"
                   r"my dad|my mom|my mother|my father|"
                   r"my cousin|my cousins|"
                   r"my neighbour|my neighbours|"
                   r"my sibling|my siblings|"
                   r"my parent|my parents|"
                   r"my uncle|my uncles|"
                   r"my aunt|my aunts|"
                   r"my mate|my mates|"
                   r"my buddy|my buddies|"
                   r"my wife|my husband|"
                   r"people|fans|listeners|singer|singers|artist|artists"
                   r")\b",
    },
]


# 2. PATHS (using config.py)

INPUT_PATH = CORPUS_PATH

RESULTS_TABLES_DIR = RESULTS_DIR / "tables"
RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_SPANS = RESULTS_TABLES_DIR / "rule_spans_with_context.csv"
OUTPUT_SUMMARY = RESULTS_TABLES_DIR / "rule_summary.csv"


# 3. APPLY RULES WITH ±3 WORD CONTEXT


rows = []
rule_counter = Counter()
review_category_map = {}
category_review_hits = defaultdict(set)

with INPUT_PATH.open() as f:
    docs = json.load(f)

for doc_id, obj in enumerate(docs):
    text = obj.get("review")
    if not isinstance(text, str):
        continue

    quote_spans = spans_in_quotes(text)
    words = text.split()
    matched_categories = set()

    for rule in RULES:
        pattern = re.compile(rule["pattern"], re.IGNORECASE)

        for match in pattern.finditer(text):

            if is_inside_any(match.start(), match.end(), quote_spans):
                continue  # skip quoted material (lyrics)

            rule_counter[rule["rule_name"]] += 1
            matched_categories.add(rule["category"])
            category_review_hits[rule["category"]].add(doc_id)

            char_start = match.start()
            word_start = len(text[:char_start].split())
            left = max(0, word_start - 3)
            right = min(len(words), word_start + 4)

            rows.append({
                "doc_id": doc_id,
                "rule_name": rule["rule_name"],
                "category": rule["category"],
                "span_text": match.group(0),
                "context": " ".join(words[left:right]),
            })

    if matched_categories:
        review_category_map[doc_id] = matched_categories


# 4. WRITE SPAN OUTPUT


with OUTPUT_SPANS.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["doc_id", "rule_name", "category", "span_text", "context"]
    )
    writer.writeheader()
    writer.writerows(rows)


# 5. SUMMARY TABLE (CSV)


multi_category_reviews = sum(
    1 for cats in review_category_map.values() if len(cats) >= 2
)

summary_rows = []

for rule in RULES:
    cat = rule["category"]
    summary_rows.append({
        "category": cat,
        "total_matches": rule_counter[cat],
        "reviews_with_match": len(category_review_hits[cat]),
        "reviews_with_2plus_categories": multi_category_reviews,
    })

with OUTPUT_SUMMARY.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "category",
            "total_matches",
            "reviews_with_match",
            "reviews_with_2plus_categories",
        ]
    )
    writer.writeheader()
    writer.writerows(summary_rows)


# 6. PRINT COUNTS


print("\nMATCH COUNTS PER RULE:")
for rule, count in rule_counter.items():
    print(f"{rule}: {count}")

print(f"\nReviews with ≥2 categories: {multi_category_reviews}")
print(f"Summary written to: {OUTPUT_SUMMARY}")
print(f"Span output written to: {OUTPUT_SPANS}")
