import torch 
import torch.nn as nn
from torch.nn import functional as F

    
# Hyperparameters

batch_size = 32 # how many independet sequences to train at once
block_size = 8 # what is the maximum context length for predictions ?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
# ---------------

torch.manual_seed(1337)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()
    
    
# here are all the unique characters that occur in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers 
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s] #takes string and outputs list of ints
decode = lambda l: "".join([itos[i] for i in l]) # takes list of integers, outputs string

# Train and Test split

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # generate small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(0, len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad() # this is a decorator, it means that the function below will not be able to change the weights of the model, we don't compute gradients when calling .backward() on the loss
def estimate_loss():
    # A function that averages up the loss over multiple batches
    out = {}
    model.eval() # Simply Embedding only Model, behaves the same in eval and train mode, but good practice
    for split in ['train', 'val']: 
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly readsoff the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # we are going to plug out the rows of this matrix and arrange them in a (b,t,c) tensor. So for character 24 we take 24th row
        
    def forward(self, idx, targets=None):
        
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C) -> look at torch docs they want (B,C, ..) shape, with C at second postition
        
        
        if targets is None: 
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T) # alternative -1 in view means infer this dimension
            loss = F.cross_entropy(logits, targets) # negtaive log likelihood loss, we expect it to be around -ln(1/65) = 4.17
            
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get prediction 
            logits, loss = self(idx)
            # focus only on the last time step, This is very simple bigram model. Although we feed in longer sequences, we only use the last token to predict the next one. We extend it later
            logits = logits[:, -1, :] # becomes (B,C)
            # apply softamx to get probabilites
            probs = F.softmax(logits, dim=-1) # (B,C)
            # sample from the distribution 
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # append sample index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
              
        return idx
    
    
model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)


#### TRAINING LOOP #####
for iter in range(max_iters):
    
    # every once in a while evaluate the loss on train and val sets.  
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
    xb, yb = get_batch("train")
        
        # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))