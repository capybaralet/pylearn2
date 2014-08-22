import numpy as np
import theano as t
from scipy.spatial.distance import cosine

class CharModel():
   def __init__(self, model, char_dict, embeddings=None, fprop=None, words=None):
      space = model.get_input_space()
      data, mask = space.make_theano_batch(batch_size=1)
      fprop_path = (data, mask)
      if type(fprop) != 'list':
          fprop = [fprop]
      for fp in fprop:
          fprop_path = fp(fprop_path)
      self.fprop = t.function([data, mask], fprop_path)
      self.words = words
      self.embeddings = embeddings
      self.char_dict = char_dict
      self.ichar_dict = {v:k for k,v in char_dict.iteritems()}

   def genEmbeddings(self, ivocab):
      self.embeddings = []
      for i in range(len(ivocab)):
          emb = self.runString(ivocab[i])
          emb = emb / np.sqrt(np.sum(emb ** 2))
          self.embeddings.append(emb)

   def arrToString(self, arr):
      return reduce(lambda x,y: x+y, arr)
      
   def stringToArr(self,string):
      arr = [self.char_dict.get(c, 0) for c in string]
      return arr

   def closest(self, vec, n):
      assert (self.embeddings is not None), "You probably need to run genEmbeddings"
      words_ = []
      dists = [cosine(vec.astype('float64'),
          self.embeddings[i].astype('float64')) for i in
          xrange(len(self.embeddings))]
      sidx = np.argsort(dists)[:n]
      return sidx
         
   def run_example(self, example):
      data = np.asarray([np.asarray([np.asarray([char])]) for char in example])
      mask = np.ones((data.shape[0], data.shape[1]), dtype='float32')    
      wordvec = self.fprop(data, mask)[0]
      wordvec = wordvec / np.sqrt(np.sum(wordvec ** 2))
      return wordvec

   def findClose(self, wordvec): 
      indices = self.closest(wordvec, 15)
      close = [self.makeWord(i) for i in indices]
      return close
    
   def runString(self, string):
      return self.run_example(self.stringToArr(string))

   def displayStringRun(self,word):
      L = self.stringToArr(word)
      Lemb = self.run_example(L)
      close = self.findClose(Lemb)
      print word, ":", close

   def displayIndexRun(self, index):
      assert (self.words is not None), "You need to give words to the model"
      close = self.findClose(self.run_example(self.words[index]))
      print self.makeWord(index), ":", close
      
   def makeWord(self, i):
      w = np.asarray(map(lambda n: self.ichar_dict[n], self.words[i]))
      return self.arrToString(w)

