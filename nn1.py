import tiktoken
import random
import functools
import sys

random.seed()
with open("corpus/krzyzacy-tom-pierwszy.txt", "r") as f:
    corpus = f.read()

ngram_length = 4

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
    for token in sequence:
        print(encoder.decode([token]), end='')
    while (len(sequence) < max_tokens):
        token = generate_token(sequence[len(sequence) - (ngram_length - 1):])
        print(encoder.decode([token]), end='')
        sys.stdout.flush()
        sequence.append(token)

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
        # normalize occurence counts to get probabilities
        total_occurences = functools.reduce(lambda x,y: x + y, [x[1] for x in candidates.items()])
        probs = [(k, v / total_occurences) for k,v in candidates.items()]

        rand = random.random()
        x = 0
        for i in range(len(probs)):
            x += probs[i][1]
            if x >= rand:
                return probs[i][0]
        return random.choice(candidates.keys)
    
starting_trigram = random.choice(ngrams)
generate(list(starting_trigram), 2048)