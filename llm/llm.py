import numpy as np
import matplotlib.pyplot as plt
import re

np.set_printoptions(precision=3, suppress=True)


# Text processing

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.split()


# Load corpus

with open("corpus.txt", "r", encoding="utf-8") as f:
    corpus = [line.strip().lower() for line in f if line.strip()]

tokenized_corpus = [tokenize(s) for s in corpus]

vocab = sorted(set(word for s in tokenized_corpus for word in s))
vocab = ["<unk>"] + vocab

token_to_id = {t: i for i, t in enumerate(vocab)}
id_to_token = {i: t for t, i in token_to_id.items()}

vocab_size = len(vocab)


# Model hyperparameters

d_model = 8
num_layers = 2
learning_rate = 0.1
epochs = 200

np.random.seed(42)


# Parameters

embeddings = np.random.randn(vocab_size, d_model)

layers = []
for _ in range(num_layers):
    layers.append({
        "W_Q": np.random.randn(d_model, d_model),
        "W_K": np.random.randn(d_model, d_model),
        "W_V": np.random.randn(d_model, d_model),
        "W_O": np.random.randn(d_model, d_model),
    })

W_out = np.random.randn(d_model, vocab_size)


# Helpers

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

def causal_attention(X, layer):
    Q = X @ layer["W_Q"]
    K = X @ layer["W_K"]
    V = X @ layer["W_V"]

    scores = Q @ K.T / np.sqrt(d_model)
    mask = np.tril(np.ones(scores.shape))
    scores[mask == 0] = -1e9

    attn = softmax(scores)
    out = (attn @ V) @ layer["W_O"]
    return out, attn


# Training (output head only)

print("\nTraining output head...\n")

for epoch in range(epochs):
    total_loss = 0.0

    for sentence in tokenized_corpus:
        for i in range(len(sentence) - 1):
            context = sentence[:i + 1]
            target = sentence[i + 1]

            ids = [token_to_id.get(t, 0) for t in context]
            X = embeddings[ids]

            for layer in layers:
                X, _ = causal_attention(X, layer)

            logits = X[-1] @ W_out
            probs = softmax(logits)

            target_id = token_to_id[target]
            loss = -np.log(probs[target_id] + 1e-9)
            total_loss += loss

            grad = probs
            grad[target_id] -= 1
            W_out -= learning_rate * np.outer(X[-1], grad)

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Loss {total_loss:.3f}")

print("\nTraining complete.\n")


# Visualization helpers

def plot_attention(attn, tokens, title):
    plt.figure(figsize=(6, 5))
    plt.imshow(attn, cmap="viridis")
    plt.colorbar()
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    plt.yticks(range(len(tokens)), tokens)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_probs(probs, k=8):
    top = np.argsort(probs)[-k:][::-1]
    labels = [id_to_token[i] for i in top]
    values = probs[top]

    plt.figure(figsize=(7, 3))
    plt.bar(labels, values)
    plt.title("Next-token probabilities")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Forward pass (verbose)

def forward_verbose(tokens):
    print("\n------------------------------------- TOKENIZATION -------------------------------------")
    print(tokens)

    ids = [token_to_id.get(t, 0) for t in tokens]
    X = embeddings[ids]

    print("\n------------------------------------- EMBEDDINGS -------------------------------------")
    print(X)

    for i, layer in enumerate(layers):
        print(f"\n------------------------------------- LAYER {i+1} -------------------------------------")
        X, attn = causal_attention(X, layer)

        print("\nAttention weights:")
        print(attn)

        plot_attention(attn, tokens, f"Layer {i+1} Attention")

        print("\nLayer output:")
        print(X)

    print("\n------------------------------------- OUTPUT HEAD -------------------------------------")
    logits = X[-1] @ W_out
    probs = softmax(logits)

    for i in np.argsort(probs)[-10:][::-1]:
        print(f"{id_to_token[i]:>10s} : {probs[i]:.4f}")

    plot_probs(probs)

    print("\nPredicted next token:", id_to_token[np.argmax(probs)])


# User loop

while True:
    question = input("\nAsk a prompt (or 'quit'): ").strip()
    if question.lower() == "quit":
        break

    tokens = tokenize(question)
    forward_verbose(tokens)
