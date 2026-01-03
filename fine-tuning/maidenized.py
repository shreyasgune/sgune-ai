import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
import os

# =====================
# SETUP
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "CUDA not available"

MODEL_PATH = "models/iron_maiden_distilled_student"
MAX_LENGTH = 128
CHUNK_WORDS = 40

# =====================
# LOAD MODEL
# =====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

# =====================
# HELPERS
# =====================
def chunk_lyrics(text, chunk_words=CHUNK_WORDS):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_words):
        chunk = " ".join(words[i:i + chunk_words])
        if len(chunk.split()) >= 20:
            chunks.append(chunk)
    return chunks

def score_chunks(chunks):
    scores = []
    for chunk in chunks:
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH
        ).to(DEVICE)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        scores.append(float(probs[1]))  # Maiden probability
    return scores

# =====================
# MAIN
# =====================
parser = argparse.ArgumentParser(description="Scan lyrics for Iron Maiden style")
parser.add_argument("file", type=str, help="Path to lyrics .txt file")
args = parser.parse_args()

if not os.path.exists(args.file):
    raise FileNotFoundError(args.file)

with open(args.file, "r", encoding="utf-8") as f:
    text = f.read()

chunks = chunk_lyrics(text)
if not chunks:
    raise ValueError("No valid lyric chunks found")

scores = score_chunks(chunks)

# =====================
# REPORT
# =====================
avg_score = sum(scores) / len(scores)
high_conf = sum(s > 0.8 for s in scores)

print("\n🎵 Iron Maiden Style Scan")
print("=" * 40)
print(f"File: {args.file}")
print(f"Chunks analyzed: {len(chunks)}")
print(f"Average Maiden probability: {avg_score:.3f}")
print(f"High-confidence Maiden chunks (>0.8): {high_conf}")

print("\nVerdict:")
if avg_score > 0.8:
    print(" SCREAM FOR ME! Its IRON MAIIIDENNNN")
elif avg_score > 0.65:
    print("Eddie is sus. This might be Maiden, but can't tell")
else:
    print("What is is pop stuff? Not Iron Maiden–style")

print("\nTop scoring chunks:")
top = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)[:3]
for i, (chunk, score) in enumerate(top, 1):
    print(f"\n[{i}] Score: {score:.3f}")
    print(chunk[:200] + ("..." if len(chunk) > 200 else ""))
