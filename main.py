import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from time import time
import math

from model import GPT, GPTConfig


#---------------------------------

class DataLoaderLite:
     def __init__(self, B, T):
          self.B,  self.T = B, T

          enc = tiktoken.get_encoding("gpt2")

          with open('input.txt', 'r') as file:
               text = file.read()

          self.tokens = torch.tensor(enc.encode(text))

          print(f"loaded {len(self.tokens)} tokens")
          print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

          self.current_position = 0

     def next_batch(self):
          B, T = self.B, self.T
          
          buf = self.tokens[self.current_position:self.current_position+B*T+1]

          X = buf[:-1].view(B, T)
          Y = buf[1:].view(B, T)

          self.current_position += B*T

          #for last batch tp resetto begining
          if self.current_position + (B*T+1) > len(self.tokens):
               self.current_position = 0

          return X, Y

if torch.cuda.is_available():
     device = "cuda"
elif torch.backends.mps.is_available():
     device = 'mps'
else:
     device = 'cpu'

warmup_iters = 10
learning_rate = 6e-4
min_lr = learning_rate * 10
max_steps = 50

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_steps - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return  + coeff * (learning_rate - min_lr)

#torch.set_float32_matmul_precision('high')

#model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig(vocab_size=50304))
#model.eval()
model.train()
model.to(device)
model = torch.compile(model)

dl = DataLoaderLite(4,1024)


#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, lr=6e-4, betas=(0.9, 0.95), device_type=device)
for step in range(50):

     lr = get_lr(step)
     for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
     t0 = time()
     X, Y = dl.next_batch()
     X, Y = X.to(device), Y.to(device)

     optimizer.zero_grad()
     with torch.autocast(device_type=device, dtype=torch.bfloat16):
          logits, loss = model(X, Y)
     loss.backward()
     norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
     optimizer.step()

     torch.cuda.synchronize()
     td = (time() - t0) 
     tokens_per_sec = (dl.B * dl.T) / td
     print(f"iteration {step=} loss={loss.item()}, dt:{td * 1000:.2f}ms, norm:{norm:.4f} tokens_per_sec:{tokens_per_sec:.2f}")

import sys; sys.exit(0)

enc = tiktoken.get_encoding("gpt2")
promt = "Hello, I'm a language model,"

@torch.no_grad()
def generate( idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        op = idx
        
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= 1024 else idx[:, -1024:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = model(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            op = torch.cat((op, idx_next), dim=1)

        op = op.tolist()
        
        return [enc.decode(i) for i in op]
num_return_seq = 5
max_length = 30
#print(model(torch.tensor([enc.encode(promt)])))

#print(generate(torch.tensor([enc.encode(promt)] * 3), 30))