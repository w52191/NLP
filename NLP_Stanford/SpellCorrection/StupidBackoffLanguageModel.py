'''
Name: StupidBackoffLanguageModel
Time: 4/22/2018
correct: 88 total: 471 accuracy: 0.186837
'''

from collections import defaultdict
import math
class StupidBackoffLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.words = defaultdict(lambda: 0)
    self.bi_words = defaultdict(lambda: 0)
    self.total = 0
    self.train(corpus)
    
  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    for sentence in corpus.corpus: #get words frequency
        pre_word = ''
        for datum in sentence.data:
            self.total += 1
            word = datum.word
            self.words[word] += 1
            if pre_word: #save bigram
                bi_word = pre_word + ',' + word
                self.bi_words[bi_word] += 1 
            pre_word = word
    self.words['UNK'] = 0
    self.words = {word: self.words[word] + 1 for word in self.words} #add-1 smoothing
    #self.bi_words['UNK'] = 1
    #self.total = len(self.words)
        #self.words = {word: self.words[word] + 1 for word in self.words} #add-1 smoothing
        #self.total += len(self.words) #add V

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0
    #print "because", self.words["the"]
    pre_word = ''
    for datum in sentence:
        #if datum not in self.words:
         #   datum = 'UNK'
        bi_datum = pre_word + ',' + datum
        if pre_word:
            if bi_datum in self.bi_words:
                score += math.log(self.bi_words[bi_datum])
                score -= math.log(self.words[pre_word])
            else:
                    if datum not in self.words:
                        datum ='UNK'
                    score += math.log(0.4)
                    score += math.log(self.words[datum])
                    score -= math.log(self.total)
                    
        pre_word = datum
        
    return score