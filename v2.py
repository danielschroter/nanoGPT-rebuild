import torch 
import torch.nn as nn
from torch.nn import functional as F

    
# Hyperparameters

batch_size = 32 # how many independet sequences to train at once
block_size = 8 # what is the maximum context length for predictions ?
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embed = 32
n_head = 4
n_layer = 4
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

class Head(nn.Module):
    """ one head of self-attention """
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size))) # its not a parameter, but we want to keep it around, so we register it as a buffer. Pytorch Naming convention.
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-1,-2) * C ** (-0.5) # (B,T,C) @ (B,C,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf')) # (B, T, T) # mask out the upper triangular part of the matrix
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        
        v = self.value(x)
        
        out = wei @ v 
        return out 
    
    
class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads*head_size, n_embed) # This is for residual skip connection. Projection is a linear combination of previous output.
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B,T,C)
        out = self.proj(out) # (B,T,C)
        return out
        

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity. FeedForward is applied to each token independently """

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), # 4 times the embedding dimension, because we want to have a lot of parameters. In paper its 512 -> 2048
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed)
        )
        
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    
    def __init__(self, n_embed, n_head):
        #n_embed: embedding dimension, n_head: number of heads 
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        
    def forward(self, x):
        # We have x + self.sa(x) and x + self.ffwd(x) in the paper, such that we have the residual skip connections here.
        x = x + self.sa(self.ln1(x)) # Layer Norms before the computation (More Common nowadays than after as in the original paper)
        x = x + self.ffwd(self.ln2(x))
        return x
        
    
    
    
    
class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        # each token directly readsoff the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) # we are going to plug out the rows of this matrix and arrange them in a (b,t,c) tensor. So for character 24 we take 24th row
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed) # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size)
        
    def forward(self, idx, targets=None):
        
        B,T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C) -> look at torch docs they want (B,C, ..) shape, with C at second postition
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C) # basically we are saying that the first token is at position 0, the second at position 1, etc.
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T, vocab_size)
        
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
            # Crop idx to the last block_size tokens. We can never have more than block size coming in, otherwise positional encoding table is running out of scope (because we have positional enc)
            idx_cond = idx[:, -block_size:] # (B, T)
            # get prediction 
            logits, loss = self(idx_cond)
            # focus only on the last time step, This is very simple bigram model. Although we feed in longer sequences, we only use the last token to predict the next one. We extend it later
            logits = logits[:, -1, :] # becomes (B,C)
            # apply softamx to get probabilites
            probs = F.softmax(logits, dim=-1) # (B,C)
            # sample from the distribution 
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # append sample index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
              
        return idx
    
    
model = BigramLanguageModel()
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