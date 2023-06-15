import tiktoken
import random
import functools

random.seed()
with open("corpus/king-james-bible.txt", "r") as f:
    corpus = f.read()

ngram_length = 3

encoder = tiktoken.encoding_for_model('gpt-4')
tokenized_corpus = encoder.encode(corpus)
vocabulary = list(set(tokenized_corpus))
ngrams = []

for i in range(len(tokenized_corpus) - 2):
    ngrams.append((tokenized_corpus[i], tokenized_corpus[i+1], tokenized_corpus[i + 2]))

def generate(sequence, max_tokens):
    while (len(sequence) < max_tokens):
        token = generate_token(sequence[len(sequence) - 2], sequence[len(sequence) - 1])
        #print(encoder.decode([token]), end='')
        sequence.append(token)
    print(encoder.decode(sequence))

def generate_token(token1, token2):
    candidates = {}
    for trigram in ngrams:
        if trigram[0] == token1:
            if trigram[1] == token2:
                if not trigram[2] in candidates:
                    candidates[trigram[2]] = 1
                else:
                    candidates[trigram[2]] = candidates[trigram[2]] + 1
    if len(candidates) == 0:
        return random.choice(vocabulary)
    else:
        list = [(k,v) for k,v in candidates.items()]
        sum = functools.reduce(lambda x,y: x + y, [x[1] for x in list])
        probs = [(k, v / sum) for k,v in candidates.items()]
        rand = random.random()
        x = 0
        for i in range(len(probs)):
            x += probs[i][1]
            if x >= rand:
                return probs[i][0]
        return random.choice(candidates.keys)
    
def encode():
    pass

def decode():
    pass
    
starting_trigram = random.choice(ngrams)
generate(list(starting_trigram), 200)