# Programming homework 1: Essay Classification with n-Gram Language Models

### **Introduction**

In this assignment you will build a **trigram** language model in Python.   
You will complete the code provided in the file <span class="ally-file-link-holder link_holder"><span class="instructure_file_holder link_holder instructure_file_link_holder">[trigram_model.py](https://courseworks2.columbia.edu/courses/168835/files/16610060?wrap=1 "trigram_model.py")[<span role="presentation"></span><span class="screenreader-only">Download trigram_model.py</span>](https://courseworks2.columbia.edu/courses/168835/files/16610060/download?download_frd=1)</span></span>. The main component of the language model will be implemented in the class TrigramModel. Parts of this class have already been provided for you and are explained below.

One important idea behind implementing language models is that the probability distributions are not precomputed. Instead, the model only stores the raw counts of n-gram occurrences and then computes the probabilities on demand. This makes smoothing possible.

The data you will work with is available in a single zip file here:<span class="ally-file-link-holder link_holder"><span class="instructure_file_holder link_holder instructure_file_link_holder">[](https://courseworks2.columbia.edu/courses/153361/files/14606739?wrap=1 "hw1_data.zip")<span class="instructure_file_holder link_holder instructure_file_link_holder">[hw1_data.zip](https://courseworks2.columbia.edu/courses/168835/files/16600695?wrap=1 "hw1_data.zip")[<span role="presentation"></span><span class="screenreader-only">Download hw1_data.zip</span>](https://courseworks2.columbia.edu/courses/168835/files/16600695/download?download_frd=1)</span> <span class="screenreader-only">.</span> </span></span>There are two data sets in this zip file, which are described below in more detail.

### **Part 1 - extracting n-grams from a sentence (20 pts)**

Complete the function<span> </span>`get_ngrams`, which takes a list of strings and an integer<span> </span>_n_<span> </span>as input, and returns padded<span> </span>_n_-grams over the list of strings. The result should be a list of Python tuples.

For example:

<pre>>>> get_ngrams(["natural","language","processing"],1)  
[('START',), ('natural',), ('language',), ('processing',), ('STOP',)]  
<span>>>> get_ngrams(["natural","language","processing"],2)  
</span><span>('START', 'natural'), ('natural', 'language'), ('language', 'processing'), ('processing', 'STOP')]  
>>> get_ngrams(["natural","language","processing"],3)  
[('START', 'START', 'natural'), ('START', 'natural', 'language'), ('natural', 'language', 'processing'), ('language', 'processing', 'STOP')]</span> </pre>

### **Part 2**<span> </span>-<span> </span>**counting n-grams in a corpus (15 pts)**

We will work with two different data sets. The first data set is the Brown corpus, which is a sample of American written English collected in the 1950s. The format of the data is a plain text file brown_train.txt, containing one sentence per line. Each sentence has already been tokenized. For this assignment, no further preprocessing is necessary.

Don't touch brown_test.txt yet. We will use this data to compute the perplexity of our language model.

**Reading the Corpus and Dealing with Unseen Words**

This part has been implemented for you and are explained in this section. Take a look at the function<span> </span>`corpus_reader`in trigram*model.py. This function takes the name of a text file as a parameter and returns a Python generator object. Generators allow you to iterate over a collection, one item at a time without ever having to represent the entire data set in a data structure (such as a list). This is a form of<span> </span>\_lazy evaluation*. You could use this function as follows:

<pre>>>> generator = corpus_reader("")  
>>> for sentence in generator:  
             print(sentence)  

['the', 'fulton', 'county', 'grand', 'jury', 'said', 'friday', 'an', 'investigation', 'of', 'atlanta', "'s", 'recent', 'primary', 'election', 'produced', '``', 'no', 'evidence', "''", 'that', 'any', 'irregularities', 'took', 'place', '.']  
['the', 'jury', 'further', 'said', 'in', 'term-end', 'presentments', 'that', 'the', 'city', 'executive', 'committee', ',', 'which', 'had', 'over-all', 'charge', 'of', 'the', 'election', ',', '``', 'deserves', 'the', 'praise', 'and', 'thanks', 'of', 'the', 'city', 'of', 'atlanta', "''", 'for', 'the', 'manner', 'in', 'which', 'the', 'election', 'was', 'conducted', '.']  
['the', 'september-october', 'term', 'jury', 'had', 'been', 'charged', 'by', 'fulton', 'superior', 'court', 'judge', 'durwood', 'pye', 'to', 'investigate', 'reports', 'of', 'possible', '``', 'irregularities', "''", 'in', 'the', 'hard-fought', 'primary', 'which', 'was', 'won', 'by', 'mayor-nominate', 'ivan', 'allen', 'jr', '&', '.']  
...</pre>

Note that iterating over this generator object works only once. After you are done, you need to create a new generator to do it again.   
As discussed in class, there are two sources of data sparseness when working with language models: Completely unseen words and unseen contexts. One way to deal with unseen words is to use a pre-defined lexicon before we extract ngrams. The function<span> </span>`corpus_reader`<span> </span>has an optional parameter lexicon, which should be a Python set containing a list of tokens in the lexicon. All tokens that are not in the lexicon will be replaced with a special "UNK" token.

Instead of pre-defining a lexicon, we collect one from the training corpus. This is the purpose of the function<span> </span>`get_lexicon(corpus)`. This function takes a corpus iterarator (as returned by<span> </span>`corpus_reader`) as a parameter and returns a set of all words that appear in the corpus more than once. The idea is that words that appear only once are so rare that they are a good stand-in for words that have not been seen at all in unseen text. You do not have to modify this function.

Now take a look at the<span> </span>`__init__`<span> </span>method of TrigramModel (the constructor). When a new TrigramModel is created, we pass in the filename of a corpus file. We then iterate through the corpus<span> </span>\_twice: _once to collect the lexicon, and once to count n-grams. You will implement the method to count n-grams in the next step.

**Counting n-grams**

Now it's your turn again. In this step, you will implement the method `count_ngrams`that should count the occurrence frequencies for ngrams in the corpus. The method already creates three instance variables of TrigramModel, which store the unigram, bigram, and trigram counts in the corpus. Each variable is a dictionary (a hash map) that maps the n-gram to its count in the corpus.   
For example, after populating these dictionaries, we want to be able to query

<pre>>>> model.trigramcounts[('START','START','the')]
5478
>>> model.bigramcounts[('START','the')]
5478
>>> model.unigramcounts[('the',)]
61428
</pre>

Where _model_ is an instance of TrigramModel that has been trained on a corpus. Note that the unigrams are represented as one-element tuples (indicated by the , in the end). Note that the actual numbers might be slightly different depending on how you set things up.

### **Part 3**<span> - **Raw n-gram probabilities (15 pts)**</span>

Write the methods `raw_trigram_probability(trigram)`,  `raw_bigram_probability(bigram)`, and `raw_unigram_probability(unigram)`.

<span>Each of these methods should return an unsmoothed probability computed from the trigram, bigram, and unigram counts. This part is easy, except that you also need to keep track of the total number of words in order to compute the unigram probabilities. </span>

### **Interlude - Generating text (OPTIONAL)**

This part is a little trickier. Write the method `generate_sentence`, which should return a list of strings, randomly generated from the raw trigram model. You need to keep track of the previous two tokens in the sequence, starting with ("START","START"). Then, to create the next word, look at all words that appeared in this context and get the raw trigram probability for each.  
Draw a random word from this distribution (think about how to do this -- I will give hints about how to draw a random value from a multinomial distribution on EdStem) and then add it to the sequence. You should stop generating words once the "STOP" token is generated. Here are some examples for how this method should behave:

<pre>model.generate_sentence()  
['the', 'last', 'tread', ',', 'mama', 'did', 'mention', 'to', 'the', 'opposing', 'sector', 'of', 'our', 'natural', 'resources', '.', 'STOP']  

<span>>>> model.generate_sentence()</span>  
['the', 'specific', 'group', 'which', 'caused', 'this', 'to', 'fundamentals', 'and', 'each', 'berated', 'the', 'other', 'resident', '.', 'STOP']</pre>

The optional t parameter of the method specifies the maximum sequence length so that no more tokens are generated if the "STOP" token is not reached before t words.

### **Part 4 - Smoothed probabilities (10 pts)**

Write the method `smoothed_trigram_probability(self, trigram)` which uses linear interpolation between the raw trigram, unigram, and bigram probabilities (see lecture for how to compute this). Set the interpolation parameters to lambda1 = lambda2 = lambda3 = 1/3\. Use the raw probability methods defined before.

### **Part 5 - Computing Sentence Probability (10 pts) **

Write the method<span> </span>`sentence_logprob(sentence)`, which returns the log probability of an entire sequence (see lecture how to compute this). Use the<span> </span>`get_ngrams`<span> </span>function to compute trigrams and the<span> </span>`smoothed_trigram_probability`method to obtain probabilities. Convert each probability into logspace using<span> </span>`math.log2`. For example:

<pre>
>>> math.log2(0.8)  
-0.3219280948873623
</pre>

Then, instead of multiplying probabilities, add the log probabilities. Regular probabilities would quickly become too small, leading to numeric issues, so we typically work with log probabilities instead.

### **Part 6 - Perplexity (15 pts)**

Write the method<span> </span>`perplexity(corpus)`, which should compute the perplexity of the model on an entire corpus.   
Corpus is a corpus iterator (as returned by the corpus*reader method).   
Recall that the perplexity is defined as<span> </span>\*\*\_2<sup>-l</sup>*\*\*, where l is defined as:

$${l=-\frac{1}{M}\sum_{i=1}^{m}\log p(x_i)}$$

Here M is the total number of words. So to compute the perplexity, sum the log probability for each sentence, and then divide by the _total number of words tokens in the corpus._ For consistency, use the base 2 logarithm.

Run the perplexity function on the test set for the Brown corpus brown_test.txt (see main section at the bottom of the Python file for how to do this). The perplexity should be less than 400\. Also try computing the perplexity on the training data (which should be a lot lower, unsurprisingly).   
This is a form of intrinsic evaluation.

### **Part 7 - Using the Model for Text Classification (15 pts)**

In this final part of the problem we will apply the trigram model to a text classification task. We will use a data set of essays written by non-native speakers of English for the ETS TOEFL test. These essays are scored according to skill level low, medium, or high. We will only consider essays that have been scored as "high" or "low". We will train a different language model on a training set of each category and then use these models to automatically score unseen essays. We compute the perplexity of each language model on each essay. The model with the lower perplexity determines the class of the essay.   
The files ets_toefl_data/train_high.txt<span> and ets_toefl_data/</span>train_low.txt in the data zip file contain the training data for high and low skill essays, respectively. The directories <span>ets_toefl_data/test_high and ets_toefl_data/test_low contain test essays (one per file) of each category. </span>

Complete the method<span> </span>`essay_scoring_experiment`. The method should be called by passing two training text files, and two testing directories (containing text files of individual essays). It returns the accuracy of the prediction.   
The method already creates two trigram models, reads in the test essays from each directory, and computes the perplexity for each essay. All you have to do is compare the perplexities and the returns the accuracy (correct predictions / total predictions).

On the essay data set, you should easily get an accuracy of > 80%.

**Data use policy**: Note that the ETS data set is proprietary and licensed to Columbia University for research and educational use only (as part of the Linguistic Data Consortium. This data set is extracted from [<span>https://catalog.ldc.upenn.edu/LDC2014T06</span>](https://catalog.ldc.upenn.edu/LDC2014T06). You may not use or share this data set for any other purpose than for this class.

## **What you need to submit**

Submit only your trigram_model.py file. Do not submit the data files.
