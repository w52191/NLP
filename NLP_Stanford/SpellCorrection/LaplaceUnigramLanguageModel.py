'''
Name: LaplaceUnigramLanguageModel
Time: 4/21/2018
correct: 52 total: 471 accuracy: 0.110403

Note: if indent line 31-33, we can get: (correct: 101 total: 471 accuracy: 0.214437)
'''

#from collections import defaultdict
import math
class LaplaceUnigramLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.words = {}
    self.total = 0
    self.train(corpus)
    
  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    for sentence in corpus.corpus: #get words frequency
        for datum in sentence.data:
            self.total += 1
            word = datum.word
            if word in self.words:
                self.words[word] += 1
            else:
                self.words[word] = 1
    self.words['UNK'] = 0
    self.words = {word: self.words[word] + 1 for word in self.words} #add-1 smoothing
    self.total += len(self.words) #add V

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0
    #print "because", self.words["the"]
    for datum in sentence:
        if datum not in self.words:
            datum = 'UNK'
        score += math.log(self.words[datum])
        score -= math.log(self.total)
        
    return score