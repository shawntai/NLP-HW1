import os
import os.path
import sys
from collections import Counter, defaultdict
import numpy as np
import math

"""
COMS W4705 - Natural Language Processing - Fall 2022 
Prorgramming Homework 1 - Trigram Language Models
Daniel Bauer
"""


def corpus_reader(corpusfile, lexicon=None):
    with open(corpusfile, 'r') as corpus:
        for line in corpus:
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon:
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else:
                    yield sequence


def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence:
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)


def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """
    ngrams = []
    sequence = ["START"] * max(1, n-1) + sequence + ["STOP"]
    for i in range(len(sequence) - n + 1):
        ngrams.append(tuple(sequence[i:i + n]))
    return ngrams


class TrigramModel(object):

    def __init__(self, corpusfile):
        # Iterate through the corpus once to build a lexicon
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")

        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """

        # might want to use defaultdict or Counter instead
        self.unigramcounts = defaultdict(int)
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)
        self.total_token_count = 0
        for sentence in corpus:
            for unigram in get_ngrams(sentence, 1):
                self.unigramcounts[unigram] += 1
            for bigram in get_ngrams(sentence, 2):
                self.bigramcounts[bigram] += 1
            for trigram in get_ngrams(sentence, 3):
                self.trigramcounts[trigram] += 1
            self.total_token_count += (len(sentence) + 1)

    def raw_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        context = trigram[:-1]
        trigram_count = self.trigramcounts[trigram]
        context_count = self.bigramcounts[context]
        if context_count > 0:
            return trigram_count / context_count
        return self.raw_unigram_probability((trigram[-1], ))

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        context = bigram[:-1]
        bigram_count = self.bigramcounts[bigram]
        context_count = self.unigramcounts[context]
        return bigram_count / context_count

    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        # hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once,
        # store in the TrigramModel instance, and then re-use it.
        return self.unigramcounts[unigram] / self.total_token_count

    def generate_next_word(self, context):
        """
        Generate the next word given a list of previous tokens.
        """
        candidates = []
        probabilities = []
        for trigram in self.trigramcounts:
            if trigram[:2] == context:
                # print('context-matching trigram:', trigram)
                candidates.append(trigram[-1])
                probabilities.append(self.raw_trigram_probability(trigram))
        # print(candidates[:10])
        # if len(candidates) == 0:
        #     print(context)
        probabilities = [p/sum(probabilities) for p in probabilities]

        return np.random.choice(candidates, p=probabilities)

    def generate_sentence(self, t=20):
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. It specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        sentence = ["START", "START",]
        while sentence[-1] != "STOP" and len(sentence) < t:
            # print('context: ', tuple(sentence[-2:]))
            sentence.append(self.generate_next_word(tuple(sentence[-2:])))
            # print('word generated: ', sentence[-1])
        return sentence

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1 / 3.0
        lambda2 = 1 / 3.0
        lambda3 = 1 / 3.0
        return lambda1 * self.raw_trigram_probability(trigram) + \
            lambda2 * self.raw_bigram_probability(trigram[1:]) + \
            lambda3 * self.raw_unigram_probability(trigram[2:])

    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigrams = get_ngrams(sentence, 3)
        return sum([math.log2(self.smoothed_trigram_probability(trigram)) for trigram in trigrams])

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        sentences_prob_sum, total_token_count = 0, 0
        for sentence in corpus:
            sentences_prob_sum += self.sentence_logprob(sentence)
            total_token_count += (len(sentence) + 1)
        return 2 ** (-1 * sentences_prob_sum / total_token_count)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):
    model1 = TrigramModel(training_file1)
    model2 = TrigramModel(training_file2)

    total = 0
    correct = 0

    for f in os.listdir(testdir1):
        pp1 = model1.perplexity(corpus_reader(
            os.path.join(testdir1, f), model1.lexicon))
        pp2 = model2.perplexity(corpus_reader(
            os.path.join(testdir1, f), model2.lexicon))
        total += 1
        if pp1 < pp2:
            correct += 1

    for f in os.listdir(testdir2):
        pp1 = model1.perplexity(corpus_reader(
            os.path.join(testdir2, f), model1.lexicon))
        pp2 = model2.perplexity(corpus_reader(
            os.path.join(testdir2, f), model2.lexicon))
        total += 1
        if pp1 > pp2:
            correct += 1

    return correct / total


if __name__ == "__main__":
    print(get_ngrams(["natural", "language", "processing"], 1))
    print(get_ngrams(["natural", "language", "processing"], 2))
    print(get_ngrams(["natural", "language", "processing"], 3))
    # cd = defaultdict(int, Counter(["hi", "hi", "hello"]))
    # print(cd['hii'])

    # model = TrigramModel(sys.argv[1])
    model = TrigramModel("hw1_data/brown_train.txt")
    # for trigram in model.trigramcounts:
    #     if trigram[0] == "START" and trigram[1] == "START":
    #         print(trigram)
    # print(model.unigramcounts)
    # print({k: v for (k, v) in model.trigramcounts.items() if v >
    #       100 and 'START' not in k and 'UNK' not in k and 'STOP' not in k})
    # print(model.trigramcounts[('START', 'START', 'the')])
    # print(model.bigramcounts[('START', 'the')])
    # print(model.unigramcounts[('the',)])

    # raw_bigramp = model.raw_bigram_probability(('START', 'the'))
    # print(raw_bigramp)
    # print(model.raw_unigram_probability(('anta',)))
    # print(model.raw_trigram_probability(('START', 'START', 'anta')))
    # print(model.trigramcounts[('of', 'the', 'united')
    #                           ] / model.bigramcounts[('of', 'the')])
    # print(model.trigramcounts[('of', 'the', 'united')])
    # print(model.bigramcounts[('of', 'the')])
    # print(model.raw_trigram_probability(('of', 'the', 'united')))
    # print(model.trigramcounts[('of', 'the', 'united')
    #                           ] / model.bigramcounts[('of', 'the')])

    # for i in range(60):
    #     print(' '.join([s for s in model.generate_sentence(t=100)[2:-1] if "UNK" not in s]) + '\n')

    # put test code here...
    # or run the script from the command line with
    # $ python -i trigram_model.py [corpus_file]
    # >>>
    #
    # you can then call methods on the model instance in the interactive
    # Python prompt.

    # Testing perplexity:
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    train_corpus = corpus_reader("hw1_data/brown_train.txt", model.lexicon)
    pp = model.perplexity(train_corpus)
    print(pp)
    dev_corpus = corpus_reader("hw1_data/brown_test.txt", model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(pp)

    # # Essay scoring experiment:
    acc = essay_scoring_experiment("hw1_data/ets_toefl_data/train_high.txt", "hw1_data/ets_toefl_data/train_low.txt",
                                   "hw1_data/ets_toefl_data/test_high", "hw1_data/ets_toefl_data/test_low")
    print(acc)
