import functools
import random
import sys
import tiktoken

NGRAM_LENGTH = 4
MAX_TOKENS = 2048

def initialize_ngrams(NGRAM_LENGTH):
    with open("corpus/krzyzacy-tom-pierwszy.txt", "r") as f:
        corpus = f.read()

    tokenized_corpus = encoder.encode(corpus)
    vocabulary = list(set(tokenized_corpus))
    ngrams = []

    for i in range(len(tokenized_corpus) -  NGRAM_LENGTH):
        ngram = []
        for j in range(NGRAM_LENGTH):
            ngram.append(tokenized_corpus[i + j])
        ngrams.append(ngram)
    return ngrams, vocabulary

def matches_ngram(ngram, tokens):
    for i in range(NGRAM_LENGTH - 1):
        if ngram[i] != tokens[i]:
            return False
    return True

def generate(starting_sequence, ngrams, vocabulary):
    for token in starting_sequence:
        yield token
    tokens = list(starting_sequence)
    while True:
        candidates = {}
        token = None
        for ngram in ngrams:
            if matches_ngram(ngram, tokens):
                ngram_tail = ngram[NGRAM_LENGTH - 1]
                if not ngram_tail in candidates:
                    candidates[ngram_tail] = 1
                else:
                    candidates[ngram_tail] = candidates[ngram_tail] + 1
        if len(candidates) == 0:
            token = random.choice(vocabulary)
        else:
            # normalize occurence counts to get probabilities
            total_occurences = functools.reduce(lambda x,y: x + y, [x[1] for x in candidates.items()])
            probs = [(k, v / total_occurences) for k,v in candidates.items()]

            rand = random.random()
            x = 0
            for i in range(len(probs)):
                x += probs[i][1]
                if x >= rand:
                    token = probs[i][0]
                    break
        tokens.pop(0)
        tokens.append(token)
        yield token

random.seed()
encoder = tiktoken.encoding_for_model('gpt-4')

ngrams, vocabulary = initialize_ngrams(NGRAM_LENGTH)
    
starting_ngram = random.choice(ngrams)
generator = generate(list(starting_ngram[:-1]), ngrams, vocabulary)
for i in range(MAX_TOKENS):
    print(encoder.decode([next(generator)]), end='')
    sys.stdout.flush()