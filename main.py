import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

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

num_return_seq = 5
max_length = 30

#model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())
model.eval()
model.to(device)

dl = DataLoaderLite(4,32)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):

     X, Y = dl.next_batch()
     X, Y = X.to(device), Y.to(device)

     optimizer.zero_grad()
     logits, loss = model(X, Y)
     print(f"iteration {i=} loss={loss.item()}")
     loss.backward()
     optimizer.step()


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
#print(model(torch.tensor([enc.encode(promt)])))

#print(generate(torch.tensor([enc.encode(promt)] * 3), 30))