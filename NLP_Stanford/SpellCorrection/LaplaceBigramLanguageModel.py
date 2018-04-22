'''
Name: LaplaceBigramLanguageModel
Time: 4/21/2018
correct: 68 total: 471 accuracy: 0.144374
'''

#from collections import defaultdict
import math
class LaplaceBigramLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.words = {}
    self.bi_words = {}
    self.total = 0
    self.train(corpus)
    
  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    for sentence in corpus.corpus: #get words frequency
        pre_word = ''
        for datum in sentence.data:
            #self.total += 1
            word = datum.word
            if word in self.words:
                self.words[word] += 1
            else:
                self.words[word] = 1
            if pre_word: #save bigram
                bi_word = pre_word + ',' + word
                if bi_word in self.bi_words:
                    self.bi_words[bi_word] += 1 
                else:
                    self.bi_words[bi_word] = 1
            pre_word = word
    self.words['UNK'] = 0
    self.bi_words['UNK'] = 0
    self.total = len(self.words)
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
        if datum not in self.words:
            datum = 'UNK'
        bi_datum = pre_word + ',' + datum
        if pre_word and bi_datum not in self.bi_words:
            bi_datum = 'UNK'
            score += math.log(self.bi_words[bi_datum]+1)
            score -= math.log(self.total+self.words[pre_word])
        pre_word = datum
        
    return score