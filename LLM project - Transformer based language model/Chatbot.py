import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse
import re

parser = argparse.ArgumentParser(description='Word-Level Transformer Chatbot')

# Hyperparameters (must match training)
block_size = 64        # Sequence length in words
n_embd = 384           # Embedding dimension
n_head = 8             # Number of attention heads
n_layer = 8            # Number of transformer layers
dropout = 0.1          # Dropout rate


# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Vocabulary and tokenization setup
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'

# Load vocabulary mappings
try:
    with open('word_to_int.pkl', 'rb') as f:
        word_to_int = pickle.load(f)
    with open('int_to_word.pkl', 'rb') as f:
        int_to_word = pickle.load(f)
    vocab_size = len(word_to_int)
    print(f"Vocabulary loaded. Size: {vocab_size}")
except FileNotFoundError:
    print("Error: Vocabulary files (word_to_int.pkl and int_to_word.pkl) not found.")
    print("These should have been created during training.")
    exit()

def encode(text_string):
    """Tokenize and encode text into word indices"""
    tokens = re.findall(r'\w+|[^\s\w]+', text_string.lower())
    return [word_to_int.get(token, word_to_int.get(UNK_TOKEN)) for token in tokens]

def decode(list_of_ints):
    """Decode word indices back to text"""
    return ' '.join([int_to_word.get(i, UNK_TOKEN) for i in list_of_ints])

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.padding_token_id = word_to_int.get(PAD_TOKEN, 0)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd, padding_idx=self.padding_token_id)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, index, targets=None):
        B, T = index.shape
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=self.padding_token_id)
        return logits, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, _ = self.forward(index_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index

# Load model
print('Loading model parameters...')
try:
    with open('model-v1.pkl', 'rb') as f:
        model = pickle.load(f)
    model = model.to(device)
    model.eval()
    print('Model loaded successfully')
except FileNotFoundError:
    print("Error: Model file 'model-v1.pkl' not found.")
    exit()

# Interactive chatbot loop
print("\nChatbot ready! Type your prompt and press Enter. Type 'quit' to exit.\n")
while True:
    prompt = input("You: ")
    if prompt.lower() == 'quit':
        break
    
    # Tokenize and encode input
    encoded = encode(prompt)
    if not encoded:
        print("Bot: (I didn't understand that input)")
        continue
        
    context = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate response
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=50)[0].tolist()
    
    # Decode and print response
    response = decode(generated[len(encoded):])  # Only show new tokens
    print(f"Bot: {response}\n")