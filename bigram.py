import torch
import random

with open('data/file.txt', 'r', encoding='utf-8') as f:
    s = f.read()

s = s.lower()
data = s.split("\n")

vocab_size = len(set(s))
chars = ''.join(sorted(list(set(s))))

int_to_chars = {i:ch for i, ch in enumerate(chars)}
int_to_chars[len(int_to_chars)] = '!'
chars_to_int = {ch:i for i, ch in enumerate(chars)}
chars_to_int['!'] = len(chars_to_int)

t = torch.zeros(vocab_size + 1, vocab_size + 1)
#print(chars_to_int)
for word in data:
    word = word + '!'
    idy = chars_to_int['!']
    for c in word:
        idx = chars_to_int[c]
        #print(idx, c)
        t[idy][idx] += 1
        idy = idx

t += 1
Prob = t / t.sum(dim=1, keepdim=True)

for i in range(10):
    w = ''
    idy = chars_to_int['!']
    while True:
        probabilities = Prob[idy]
        i = torch.multinomial(probabilities, 1)
        c = int_to_chars[i.item()]
        w += c
        if c == '!':
            break

        idy = chars_to_int[c]

    print(w)

