import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from time import time
import math
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import os

from model import GPT, GPTConfig


#---------------------------------

class DataLoaderLite:
     def __init__(self, B, T, process_rank, num_processes):
          self.B,  self.T = B, T
          self.process_rank = process_rank
          self.num_processes = num_processes

          enc = tiktoken.get_encoding("gpt2")

          with open('input.txt', 'r') as file:
               text = file.read()

          self.tokens = torch.tensor(enc.encode(text))

          print(f"loaded {len(self.tokens)} tokens")
          print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

          self.current_position = self.B * self.T * self.process_rank

     def next_batch(self):
          B, T = self.B, self.T
          
          buf = self.tokens[self.current_position:self.current_position+B*T+1]

          X = buf[:-1].view(B, T)
          Y = buf[1:].view(B, T)

          self.current_position += B * T * self.num_processes

          #for last batch tp resetto begining
          if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
               self.current_position = self.B * self.T * self.process_rank

          return X, Y

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


ddp = int(os.environ.get('RANK', -1)) != -1 
if ddp :
     assert torch.cuda.is_available(), "could not run on cpu"
     init_process_group(backend='nccl')

     ddp_rank = int(os.environ['RANK'])
     ddp_local_rank = int(os.environ['LOCAL_RANK'])
     ddp_world_size = int(os.environ['WORLD_SIZE'])
     device = f'cuda:{ddp_local_rank}'

     torch.cuda.set_device(device=device)
     master_process = ddp_rank == 0
else:
     ddp_rank = 0
     ddp_local_rank = 0
     ddp_world_size = 1
     master_process =True
     device = 'cpu'
     if torch.cuda.is_available():
          device = "cuda"
     elif torch.backends.mps.is_available():
          device = 'mps'
     

dtype = 'float16'
warmup_iters = 10
learning_rate = 6e-4
min_lr = learning_rate * 0.1
max_steps = 5


ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device, dtype=ptdtype)  if device == 'cuda' else nullcontext()

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)
if ddp :
     model = DDP(model, device_ids=[ddp_local_rank])

total_batch_size = 524288 # 2 ** 19 
B = 16
T = 1024

assert total_batch_size % (B * T * ddp_world_size) == 0, f"make {total_batch_size=} divisible by {B=} * {T=} * {ddp_world_size=}"
grad_accum_steps = total_batch_size // ( B * T * ddp_world_size)

if master_process:
     print(f"total desird batch size: {total_batch_size}")
     print(f"=> caluclated gradient accumulation steps: {grad_accum_steps}")



dl = DataLoaderLite(B, T, ddp_local_rank, ddp_world_size)


#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, betas=(0.9, 0.95), device_type=device)
# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(device=device, enabled=(dtype == 'float16'))

for step in range(20):

     lr = get_lr(step)
     for param_group in optimizer.param_groups:
        param_group['lr'] = lr

     t0 = time()
     loss_accum = 0.0
     for micro_step in range(grad_accum_steps):
          X, Y = dl.next_batch()
          X, Y = X.to(device), Y.to(device)
          with ctx:
               logits, loss = model(X, Y)
               loss = loss / grad_accum_steps
          loss_accum += loss.detach()

          if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

          scaler.scale(loss).backward()

     if ddp:
          dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

     scaler.unscale_(optimizer)
     norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
     # step the optimizer and scaler if training in fp16
     scaler.step(optimizer)
     scaler.update()

     # flush the gradients as soon as we can, no need for this memory anymore
     optimizer.zero_grad(set_to_none=True)

     if device == 'cuda' : 
          torch.cuda.synchronize() 
     elif device == 'mps' :
          torch.mps.synchronize() 

     td = (time() - t0) 
     tokens_per_sec = (dl.B * dl.T * grad_accum_steps * ddp_world_size) / td

     if master_process:
          print(f"iteration {step=} |  loss_acc : {loss_accum:.4f} | dt:{td * 1000:.2f}ms | norm:{norm:.4f} | tokens_per_sec:{tokens_per_sec:.2f} |")


if ddp: destroy_process_group()

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