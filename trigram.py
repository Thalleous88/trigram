words = open('names.txt', 'r').read().splitlines()

import torch
t = {}

for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
        trigram = (ch1, ch2, ch3)
        t[trigram] = t.get(trigram, 0) + 1
t
        
sorted(t.items(), key = lambda kv: kv[1], reverse=True)
len(t)
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
stoi
N = torch.zeros((27, 27, 27), dtype=torch.int32)
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        ix3 = stoi[ch3]
        N[ix1, ix2, ix3] += 1
        
N[0:1:]
itos = {i:s for s,i in stoi.items()}
itos
N[2, 0, :]
# Assume N is your trigram count tensor
P = N.float()

row_sums = P.sum(2, keepdim=True)
row_sums[row_sums == 0] = 1  # to avoid division by zero
P = P / row_sums


g = torch.Generator().manual_seed(2147483647)

for i in range(15):
    out = []
    ix1, ix2 = 0, 0  # Start with (0, 0) bigram
    while True:
        p = P[ix1][ix2]  # Get probability distribution over next char

        # Handle the case where p.sum() == 0
        if p.sum() == 0:
            p = torch.ones_like(p) / len(p)

        ix3 = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix3])
        if ix3 == 0:
            break

        # Move forward: (ix1, ix2) -> (ix2, ix3)
        ix1, ix2 = ix2, ix3

    print(''.join(out))

P = N.float()
row_sums = P.sum(1, keepdim=True)
# Avoid division by zero
row_sums[row_sums == 0] = 1
P = P / row_sums
P[0][2].sum()
N = torch.zeros((27, 27, 27), dtype=torch.int32)
N.shape
# create the training set of trigram

xs, ys = [], []

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        ix3 = stoi[ch3]
        N[ix1, ix2, ix3] += 1
        xs.append((ix1, ix2))
        ys.append(ix3)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
import torch.nn.functional as F

# Split xs into two parts: first and second characters
x1 = F.one_hot(xs[:, 0], num_classes=27).float()  # shape: [N, vocab_size]
x2 = F.one_hot(xs[:, 1], num_classes=27).float()  # shape: [N, vocab_size]

# Concatenate the one-hot vectors for each pair
xenc = torch.cat([x1, x2], dim=1).float()  # shape: [N, vocab_size * 2]
xenc.shape
g = torch.Generator().manual_seed(2147483647+3)
W = torch.randn((54, 27), generator=g, requires_grad = True)
W
logits = (xenc @ W) #log-counts
counts = logits.exp()
probs = counts/counts.sum(1, keepdims=True)
probs
probs[0]
probs.shape
for k in range(100):
    # Split xs into two parts: first and second characters
    x1 = F.one_hot(xs[:, 0], num_classes=27).float()  # shape: [N, vocab_size]
    x2 = F.one_hot(xs[:, 1], num_classes=27).float()  # shape: [N, vocab_size]

    # Concatenate the one-hot vectors for each pair
    xenc = torch.cat([x1, x2], dim=1).float()  # shape: [N, vocab_size * 2]
    logits = xenc @ W # log-counts, W here is 54 x 27 because it predicts 27 set of characters
    counts = logits.exp() # counts, equivalent to N (softmax function)
    probs = counts/counts.sum(1, keepdims=True) # probabilities for the next character
    loss = -probs[torch.arange(xs.shape[0]), ys].log().mean() + 0.01*(W**2).mean()
    print(loss.item())

    W.grad = None
    loss.backward()

    W.data += -50 * W.grad
    
g = torch.Generator().manual_seed(2147483647)

for i in range(15):
    out = []
    ix1, ix2 = 0, 0

    while True:
        xenc = F.one_hot(torch.tensor([ix1, ix2]), num_classes=27).view(-1)  # Flatten the one-hot vector
        xenc = xenc.float()  # Convert to float tensor
        logits = xenc @ W # log-counts
        counts = logits.exp() # counts, equivalent to N (softmax function)
        p = counts/counts.sum(0, keepdims=True) # probabilities for the next 
       
        ix3 = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix3])
        if (ix3 == 0): break

        ix1, ix2 = ix2, ix3

    print(''.join(out))
    
# Assume N is your trigram count tensor
P = N.float()

row_sums = P.sum(2, keepdim=True)
row_sums[row_sums == 0] = 1  # to avoid division by zero
P = P / row_sums


g = torch.Generator().manual_seed(2147483647)

for i in range(15):
    out = []
    ix1, ix2 = 0, 0  # Start with (0, 0) bigram
    while True:
        p = P[ix1][ix2]  # Get probability distribution over next char

        # Handle the case where p.sum() == 0
        if p.sum() == 0:
            p = torch.ones_like(p) / len(p)

        ix3 = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix3])
        if ix3 == 0:
            break

        # Move forward: (ix1, ix2) -> (ix2, ix3)
        ix1, ix2 = ix2, ix3

    print(''.join(out))
