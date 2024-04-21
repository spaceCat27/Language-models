import torch
import random
import torch.nn.functional as F


with open('data/file.txt', 'r', encoding='utf-8') as f:
    data = f.read()

data = data.lower()
chars = set(''.join(data.splitlines()))
chars_num = len(chars) + 1
words = data.splitlines() 

itos = {i:ch for i, ch in enumerate(chars)}
itos[len(itos)] = '!'
stoi = {ch:i for i, ch in enumerate(chars)}
stoi['!'] = len(stoi) 

block_size = 8
def build_data(data):
    X, y = [], []
    
    for w in data:
        sample = [stoi['!']] * block_size
        for ch1, ch2 in zip('!'+w, w+'!'):
            
            sample = sample[1:] + [stoi[ch1]]
            X.append(sample)
            y.append(stoi[ch2])

    X = torch.tensor(X)
    y = torch.tensor(y)

    return X, y


random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, ytr = build_data(words[:n1])
Xval, yval = build_data(words[n1:n2])
Xtst, ytst = build_data(words[n2:])

feature_n = 24
neurons_n = 128
feature_table = torch.randn(chars_num, feature_n)
W1 = torch.randn(neurons_n, feature_n * 2) * 0.1
b1 = torch.zeros(1, neurons_n)
W2 = torch.randn(neurons_n, neurons_n * 2) * 0.1
b2 = torch.zeros(1, neurons_n)
W3 = torch.randn(chars_num, neurons_n * 2) * 0.1
b3 = torch.zeros(1, chars_num)


parameters = [feature_table, W1, b1, W2, b2, W3, b3]
for p in parameters:
    p.requires_grad = True

#print(Xtr.shape[0])
epochs = 200_001
lr = 0.01
batch_size = 32
l = []
for e in range(epochs):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    emb = feature_table[Xtr[ix]]    
    l1 = emb.view(batch_size, -1, feature_n * 2) @ W1.T + b1
    l1 = l1.view(batch_size, -1, neurons_n * 2)
    h1 = torch.tanh(l1)
    l2 = h1 @ W2.T + b2
    l2 = l2.view(batch_size, neurons_n * 2)
    h2 = torch.tanh(l2)
    l3 = h2 @ W3.T + b3
    loss = F.cross_entropy(l3, ytr[ix])
    if e % 200 == 0:
        print(f'{e}: {loss.item()}')
    
    l.append(loss.item())
    for p in parameters:
        p.grad = None

    loss.backward()

    for p in parameters:
        p.data -= lr * p.grad
    

for i in range(10):

    word = ''
    sample = [stoi['!']] * block_size
    while True:
        emb = feature_table[sample]    
        l1 = emb.view(1, -1, feature_n * 2) @ W1.T + b1
        h1 = torch.tanh(l1.view(1, -1, neurons_n * 2))
        l2 = h1 @ W2.T + b2
        h2 = torch.tanh(l2.view(1, neurons_n * 2))
        l3 = h2 @ W3.T + b3
        probs = torch.softmax(l3, dim=1)

        ch = torch.multinomial(probs, num_samples=1, replacement=True).item()
        word = word + itos[ch]
        sample = sample[1:] + [ch] 

        if itos[ch] == '!':
            break
    
    print(f'{i}: {word}')


