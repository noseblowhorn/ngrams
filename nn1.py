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

for i in range(len(tokenized_corpus) -  ngram_length):
    ngram = []
    for j in range(ngram_length):
        ngram.append(tokenized_corpus[i + j])
    ngrams.append(ngram)

def generate(sequence, max_tokens):
    while (len(sequence) < max_tokens):
        token = generate_token(sequence[len(sequence) - (ngram_length - 1):])
        sequence.append(token)
    print(encoder.decode(sequence))

def matches_ngram(ngram, tokens):
    for i in range(ngram_length - 1):
        if ngram[i] != tokens[i]:
            return False
    return True

def generate_token(tokens):
    candidates = {}
    for ngram in ngrams:
        if matches_ngram(ngram, tokens):
            ngram_tail = ngram[ngram_length - 1]
            if not ngram_tail in candidates:
                candidates[ngram_tail] = 1
            else:
                candidates[ngram_tail] = candidates[ngram_tail] + 1
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