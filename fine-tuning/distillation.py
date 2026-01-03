import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from datasets import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# =====================
# CUDA CHECK
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "❌ CUDA not available. Check your PyTorch install."

# =====================
# DIRECTORIES
# =====================
os.makedirs("models", exist_ok=True)
os.makedirs("graphs", exist_ok=True)

# =====================
# CONFIG
# =====================
TEACHER_MODEL = "bert-base-uncased"       # Steve Harris
STUDENT_MODEL = "prajjwal1/bert-tiny"     # Touring bassist

MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 10
LR = 2e-5

TEMPERATURE = 3.0
ALPHA = 0.7    # supervised
BETA = 0.3     # distillation

# =====================
# LOAD LYRICS (LOCAL)
# =====================
def load_lyrics(folder, label):
    samples = []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # basic cleanup
        text = text.replace("[Chorus]", "").replace("[Verse]", "")
        lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 20]

        # chunk into small lyric segments
        chunk = ""
        for line in lines:
            chunk += " " + line
            if len(chunk.split()) >= 40:
                samples.append({"text": chunk.strip(), "label": label})
                chunk = ""

    return samples

maiden_samples = load_lyrics("data/iron_maiden", label=1)
non_maiden_samples = load_lyrics("data/non_maiden", label=0)

dataset = Dataset.from_list(maiden_samples + non_maiden_samples)
dataset = dataset.shuffle(seed=42)

print(f"Loaded {len(dataset)} lyric samples")

# =====================
# TOKENIZATION
# =====================
tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =====================
# MODELS (SAFE LOADING)
# =====================
teacher = AutoModelForSequenceClassification.from_pretrained(
    TEACHER_MODEL,
    num_labels=2,
    dtype=torch.float32,
    use_safetensors=True
).to(DEVICE)

teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

student = AutoModelForSequenceClassification.from_pretrained(
    STUDENT_MODEL,
    num_labels=2,
    dtype=torch.float32,
    use_safetensors=True
).to(DEVICE)

optimizer = AdamW(student.parameters(), lr=LR)


# =====================
# LOSSES
# =====================
ce_loss = nn.CrossEntropyLoss()
kl_loss = nn.KLDivLoss(reduction="batchmean")


# =====================
# LOGGING
# =====================
total_losses = []
sup_losses = []
distill_losses = []
teacher_entropy = []
student_entropy = []

# =====================
# TRAINING LOOP
# =====================
for epoch in range(EPOCHS):
    student.train()
    bar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for batch in bar:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        with torch.no_grad():
            t_logits = teacher(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).logits

        s_logits = student(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).logits

        loss_sup = ce_loss(s_logits, labels)

        loss_distill = kl_loss(
            nn.functional.log_softmax(s_logits / TEMPERATURE, dim=-1),
            nn.functional.softmax(t_logits / TEMPERATURE, dim=-1)
        ) * (TEMPERATURE ** 2)

        loss = ALPHA * loss_sup + BETA * loss_distill
        loss.backward()
        optimizer.step()

        # logging
        total_losses.append(loss.item())
        sup_losses.append(loss_sup.item())
        distill_losses.append(loss_distill.item())

        t_prob = torch.softmax(t_logits, dim=-1)
        s_prob = torch.softmax(s_logits, dim=-1)

        teacher_entropy.append(
            (-t_prob * torch.log(t_prob + 1e-8)).sum(dim=-1).mean().item()
        )
        student_entropy.append(
            (-s_prob * torch.log(s_prob + 1e-8)).sum(dim=-1).mean().item()
        )

        bar.set_postfix(
            loss=f"{loss.item():.3f}",
            sup=f"{loss_sup.item():.3f}",
            dist=f"{loss_distill.item():.3f}"
        )

# =====================
# SAVE MODEL
# =====================
student.save_pretrained("models/iron_maiden_distilled_student")
tokenizer.save_pretrained("models/iron_maiden_distilled_student")

# =====================
# PLOTS
# =====================
plt.figure()
plt.plot(total_losses)
plt.title("Total Training Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.savefig("graphs/maiden_total_loss.png")
plt.close()

plt.figure()
plt.plot(sup_losses, label="Supervised")
plt.plot(distill_losses, label="Distillation")
plt.legend()
plt.title("Loss Components")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.savefig("graphs/maiden_loss_components.png")
plt.close()

plt.figure()
plt.plot(teacher_entropy, label="Teacher")
plt.plot(student_entropy, label="Student")
plt.legend()
plt.title("Prediction Entropy (Style Confidence)")
plt.xlabel("Step")
plt.ylabel("Entropy")
plt.savefig("graphs/maiden_entropy.png")
plt.close()

print("Training complete")
print("Model saved to models/iron_maiden_distilled_student")
print("Graphs saved to graphs/")
