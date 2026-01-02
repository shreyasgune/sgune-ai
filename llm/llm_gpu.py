import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# ***********************
# SAFETY & DIRECTORIES
# ***********************

assert torch.cuda.is_available(), "CUDA NOT DETECTED — check drivers!"
torch.cuda.set_device(0)
device = "cuda"

os.makedirs("graphs", exist_ok=True)
os.makedirs("model", exist_ok=True)

# ***********************
# LOAD CORPUS
# ***********************

with open("maiden_corpus.txt", "r", encoding="utf-8") as f:
    corpus = [line.strip() for line in f if line.strip()]

# ***********************
# BPE TOKENIZER (GPT STYLE)
# ***********************

tokenizer_path = "model/tokenizer.json"

if os.path.exists(tokenizer_path):
    tokenizer = Tokenizer.from_file(tokenizer_path)
else:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=8000,
        min_frequency=2,
        special_tokens=["<pad>", "<unk>"]
    )

    tokenizer.train_from_iterator(corpus, trainer)
    tokenizer.save(tokenizer_path)

vocab_size = tokenizer.get_vocab_size()
pad_id = tokenizer.token_to_id("<pad>")

# ***********************
# HYPERPARAMETERS
# ***********************

d_model = 128
num_heads = 4
num_layers = 4
mlp_ratio = 4
max_len = 128

lr = 3e-4
epochs = 200

# ***********************
# DATASET
# ***********************

def make_dataset(texts):
    X, Y = [], []
    for t in texts:
        ids = tokenizer.encode(t).ids[:max_len]
        if len(ids) < 2:
            continue
        X.append(ids[:-1])
        Y.append(ids[1:])
    return X, Y

X_data, Y_data = make_dataset(corpus)

# ***********************
# MODEL
# ***********************

class MultiHeadCausalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, return_attn=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(B, T, num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, device=device))
        scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.out(out)

        return (out, attn) if return_attn else out

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadCausalAttention()
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model)
        )

    def forward(self, x, return_attn=False):
        if return_attn:
            a, attn = self.attn(self.ln1(x), True)
            x = x + a
            x = x + self.mlp(self.ln2(x))
            return x, attn
        else:
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x

class GlassGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([Block() for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, return_attn=False):
        B, T = idx.shape
        pos = torch.arange(T, device=device)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        attns = []
        for block in self.blocks:
            if return_attn:
                x, attn = block(x, True)
                attns.append(attn)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        return (logits, attns) if return_attn else logits

model = GlassGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

ckpt_path = "model/model.pt"
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["opt"])
    print("Loaded checkpoint")

model.train()
for epoch in range(epochs):
    total_loss = 0.0
    for x, y in tqdm(zip(X_data, Y_data), total=len(X_data)):
        x = torch.tensor(x, device=device).unsqueeze(0)
        y = torch.tensor([y], device=device)

        logits = model(x)              # (B, T, vocab)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            y.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch:03d} | Loss {total_loss:.3f}")

    torch.save(
        {"model": model.state_dict(), "opt": optimizer.state_dict()},
        ckpt_path
    )

def save_attention(attn, tokens, layer, tag):
    for h in range(attn.shape[1]):
        plt.figure(figsize=(5,4))
        plt.imshow(attn[0,h].detach().cpu())
        plt.xticks(range(len(tokens)), tokens, rotation=45)
        plt.yticks(range(len(tokens)), tokens)
        plt.title(f"L{layer+1} H{h+1}")
        plt.tight_layout()
        plt.savefig(f"graphs/attn_{tag}_l{layer+1}_h{h+1}.png")
        plt.close()

def save_gradients(tag):
    grads = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    plt.plot(grads)
    plt.title("Gradient norms")
    plt.savefig(f"graphs/grads_{tag}.png")
    plt.close()

def save_weights(tag):
    w = torch.cat([p.flatten() for p in model.parameters()])
    plt.hist(w.detach().cpu(), bins=100)
    plt.title("Weight distribution")
    plt.savefig(f"graphs/weights_{tag}.png")
    plt.close()

def sample_next(logits, ids, temp=0.6, top_k=30, rep_penalty=1.3):
    logits = logits / temp

    for token in set(ids[0].tolist()):
        logits[0, token] /= rep_penalty

    k = min(top_k, logits.size(-1))
    if k > 0:
        v, _ = torch.topk(logits, k)
        logits[logits < v[:, [-1]]] = -1e9

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)



def penalize_repetition(logits, ids, penalty=1.2):
    for token in ids[0].tolist():
        logits[0, token] /= penalty
    return logits

@torch.no_grad()
def generate(prompt, steps=50):
    model.eval()
    ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)

    for _ in tqdm(range(steps)):
        logits = model(ids[:, -max_len:])[:, -1]
        probs = F.softmax(logits, dim=-1)
        next_id = sample_next(logits,ids)
        # next_id = torch.multinomial(probs, 1)
        ids = torch.cat([ids, next_id], dim=1)

    return tokenizer.decode(ids[0].tolist())

while True:
    prompt = input("\nPrompt (or quit): ")
    if prompt == "quit":
        break

    ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
    logits, attns = model(ids, return_attn=True)

    tokens = tokenizer.encode(prompt).tokens
    tag = prompt.replace(" ", "_")[:30]

    for i, attn in enumerate(attns):
        save_attention(attn, tokens, i, tag)

    save_gradients(tag)
    save_weights(tag)

    print("\nGenerated text:\n")
    print (generate(prompt))
    # print(generate(prompt, steps=50, temp=0.7, top_k=40))

