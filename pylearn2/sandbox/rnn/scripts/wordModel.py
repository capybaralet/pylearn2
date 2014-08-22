import numpy as np
import theano as t
from scipy.spatial.distance import cosine

class WordModel():
   def __init__(self, model, word_dict, embeddings):
      self.embeddings = embeddings
      self.word_dict = word_dict
      self.iword_dict = {v:k for k,v in word_dict.iteritems()}

   def closest(self, vec, n):
      words_ = []
      dists = [(cosine(vec, self.embeddings[i]), i) for i in range(30000)]
      for k in range(n):
         index = min(dists)[1]
         dists[index] = (float("inf"),index)
         words_.append(index)
      return words_
         
   def findClose(self, wordvec): 
      indices = self.closest(wordvec, 15)
      close = [self.makeWord(i) for i in indices]
      return close

   def runIndex(self, i):
      return self.embeddings[i]
    
   def runString(self, string):
      return self.runIndex(self.word_dict[string])

   def displayStringRun(self,word):
      close = self.findClose(self.runString(word))
      print word, ":", close

   def displayIndexRun(self, index):
      close = self.findClose(self.runIndex(index))
      print self.makeWord(index), ":", close
      
   def makeWord(self, i):
      w = self.iword_dict[i]
      return w

