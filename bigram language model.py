import os
import lzma
import PyPDF2
from tqdm import tqdm
import re

#Function to clean text by keeping only alphanumeric characters, spaces, and basic punctuation
def clean_text(input_text):
    # Keep only A-Z, a-z, 0-9, spaces, and basic punctuation (.,!?;:'"-)
    pattern = r'[^A-Za-z0-9\s.,!?;:\'\"-]'
    cleaned_text = re.sub(pattern, '', input_text)
    # Normalize multiple spaces to a single space and strip leading/trailing spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

# Extract and save all PDF text into a single txt file
pdf_folder = "/Users/sujangauchan/Desktop/Nutrition data/untitled folder"
output_folder = "/Users/sujangauchan/Desktop/Nutrition data"
output_file = os.path.join(output_folder, "Vitamin and minerals requirements.txt")

all_text = ""

for root, _, files in os.walk(pdf_folder):
    for file in tqdm(files, desc="Processing PDFs"):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(root, file)
            try:
                with open(pdf_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        all_text += page.extract_text() or ""
            except Exception as e:
                print(f"Failed to process {file}: {e}")

#Clean the combined text before saving
cleaned_text = clean_text(all_text)

# Save combined text to a single file
with open(output_file, "w", encoding="utf-8") as f:
    f.write(all_text)

#Importing libraries
import torch
import torch.nn as nn
from torch.nn import functional as F
device= 'mps' if torch.mps.is_available() else 'cpu'
print(device)

# #Setting Hyperparameters
# block_size=8
# batch_size=4
# max_iters= 30000
# learning_rate = 3e-4
# eval_iters = 250
# dropout = 0.2

#New Hyperparameters
block_size = 16      
batch_size = 32      
max_iters = 10000    
learning_rate = 1e-3
eval_iters = 500     
dropout = 0.1  


with open(output_file, 'r', encoding = 'utf-8') as f:
    text = f.read()
chars = sorted(set(text))
vocab_size = len(chars)
print(len(chars))

#Checking number of words in the final file
with open(output_file, 'r', encoding='utf-8') as f:
    text = f.read()
    words = text.split()  # Splits by whitespace (including spaces, tabs, newlines)
    word_count = len(words)
    print("Number of words in the file:", word_count)

# Create mapping between characters and integers
string_to_int = { ch:i for i,ch in enumerate(chars)}
int_to_string = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

# Convert text to tensor of integers
data = torch.tensor(encode(text), dtype=torch.long)
print(data[:100])

# Split data into training and validation sets (80/20 split)
n = int(0.8*len(data))
train = data[:n]
val = data[n:]

# Function to get randomized training or validation batches
def get_batch(split):
    data = train if split == 'train' else val
    ix = torch.randint(len(data) - block_size, (batch_size,)) 
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device) , y.to(device)
    return x, y

# Get a sample batch
x, y = get_batch('train')
print('inputs:')
print(x)
print('targets:')
print(y)


# Demonstrate how inputs map to targets with a concrete example
x= train[:block_size]
y= train[1:block_size+1]

for i in range(block_size):
    context= x[:i+1]
    target = y[i]
    print('when input is', context, 'target is', target)

# Function to estimate loss on train and validation sets
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k]= loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Define the Bigram Language Model
class BigramLanguage(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, index, targets = None):
        logits =self.token_embedding_table(index)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self.forward(index)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index

# Initialize the model and move it to the device
model = BigramLanguage(vocab_size)
m = model.to(device)

# Generate text before training to see random output
context = torch.zeros((1,1), dtype=torch.long, device=device)
generated_chars = decode(m.generate(context, max_new_tokens = 500)[0].tolist())
print(generated_chars)

# Import visualization libraries
import matplotlib.pyplot as plt

# Lists to store loss values for plotting
train_losses = []
val_losses = []
iterations = []

# Initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr= learning_rate)

# Training loop
for iter in range(max_iters):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step{iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")
        
        # Store loss values for plotting
        train_losses.append(losses['train'].item())
        val_losses.append(losses['val'].item())
        iterations.append(iter)

    xb, yb = get_batch('train')
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()

# Print final loss
print(loss.item())

# Generate text after training to see improved output
context = torch.zeros((1,1), dtype=torch.long, device=device)
generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(generated_chars)

# Create loss plot
plt.figure(figsize=(10, 6))
plt.plot(iterations, train_losses, 'b-o', label='Training Loss')
plt.plot(iterations, val_losses, 'r-o', label='Validation Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Bigram Model Training and Validation Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)


plt.annotate(f'Start: {train_losses[0]:.3f}', 
             xy=(iterations[0], train_losses[0]), 
             xytext=(iterations[0]+10, train_losses[0]+0.03),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=10)

plt.annotate(f'End: {train_losses[-1]:.3f}', 
             xy=(iterations[-1], train_losses[-1]), 
             xytext=(iterations[-1]-100, train_losses[-1]+0.1),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

# Save the plot
plt.savefig('bigram_model_loss.png')
plt.show()

# Additional visualization: Model improvement comparison
plt.figure(figsize=(10, 5))

# Calculate character frequency in generated text before and after training
def get_char_freq(text, top_n=10):
    freq = {}
    for c in text:
        if c in freq:
            freq[c] += 1
        else:
            freq[c] = 1
    
    # Sort by frequency
    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return sorted_freq[:top_n]

# Generate new text samples for comparison
context = torch.zeros((1,1), dtype=torch.long, device=device)
final_generated = decode(m.generate(context, max_new_tokens=200)[0].tolist())

# Print sample of final generated text
print("\nSample of generated text after training:")
print(final_generated[:100])

# Print some statistics
print("\nTraining started with loss: {:.3f}".format(train_losses[0]))
print("Training ended with loss: {:.3f}".format(train_losses[-1]))
print("Improvement: {:.2f}%".format(100 * (train_losses[0] - train_losses[-1]) / train_losses[0]))