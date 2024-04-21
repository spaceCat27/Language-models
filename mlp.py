import torch
import torch.nn.functional as F

with open('data/file.txt', 'r', encoding='utf-8') as f:
    data = f.read()

data = data.lower()
words = data.split('\n')
data = ''.join(words)

chars_num = len(set(data)) + 1
chars = list(sorted(set(data)))

itos = {i:ch for i,ch in enumerate(chars)}
stoi = {ch:i for i,ch in enumerate(chars)}
itos[len(itos)] = '!'
stoi['!'] = len(stoi)

features_n = 10
block_size = 3

Features = torch.randn(chars_num, features_n) * 0.1
W1 = torch.randn(300, features_n * block_size) * 0.1
b1 = torch.zeros(1, 300)
W2 = torch.randn(chars_num, 300) * 0.1
b2 = torch.zeros(1, chars_num)

parameter = [Features, W1, b1, W2, b2]

for p in parameter:
    p.requires_grad = True


X, y = [], []

for word in words:
    
    idy = stoi['!']
    sample = [idy] * block_size

    for ch1, ch2 in zip('!'+word, word + '!'):
        ch1 = stoi[ch1]
        ch2 = stoi[ch2]
        sample = sample[1:] + [ch1]
        X.append(sample)
        y.append(ch2)

X = torch.tensor(X)
y = torch.tensor(y)


epochs = 200_000
batch_size = 64
lr = 0.01

for e in range(epochs): 
    ind = torch.randint(0, len(X), (batch_size,))
    target = y[ind]
    emb = Features[X[ind]]
    batch = emb.view(-1, features_n * block_size).float()
    
    l1 = batch @ W1.T + b1
    t1 = torch.tanh(l1)
    l2 = t1 @ W2.T + b2
    loss = F.cross_entropy(l2, target)

    if e % 200 == 0:
        print(f'{e} loss: {loss}')
    for p in parameter:
        p.grad = None

    loss.backward()

    for p in parameter:
        p.data -= lr * p.grad


for i in range(25): 
    idy = stoi['!']
    sample = [idy] * block_size
    word = ''

    while True:
        
        emb = Features[sample]
        batch = emb.view(-1, features_n * block_size).float()
        
        l1 = batch @ W1.T + b1
        t1 = torch.tanh(l1)
        l2 = t1 @ W2.T + b2
        logits_exp = l2.exp() 
        softmax = logits_exp / logits_exp.sum(dim=1, keepdims=True)
        
        ch = torch.multinomial(softmax, 1).item()
        sample = sample[1:] + [ch]
        ch = itos[ch]
        word += ch

        if ch == '!':
            break

    print(f'{i}: {word}')

