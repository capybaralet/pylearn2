import cPickle
import numpy as np
from charModel import CharModel
from wordModel import WordModel

model_path = '../pkls/full_vocabrnn_2tanh.pkl' #'../pkls/full_vocabrnnLEAKY.pkl' 
#model_path = '../pkls/rnn_realskipgram_factored_schwenk_256_300_Ada.pkl'
chars_path = '/data/lisatmp3/devincol/data/translation_char_vocab.en.pkl'
vocab_path = '/data/lisatmp3/chokyun/mt/vocab.30k/bitexts.selected/vocab.en.pkl'
words_path = '/data/lisatmp3/devincol/data/translation_vocab_aschar.en.pkl'
embeddings_path = '/data/lisatmp3/devincol/embeddings/deepoutput_rnnSkipgram300.pkl'
#embeddings_path = '/data/lisatmp3/devincol/embeddings/skipgram300.pkl'

print "Loading Data"
with open(vocab_path) as f:
   vocab = cPickle.load(f)
ivocab = {v:k for k,v in vocab.iteritems()}
   
with open(model_path) as f:
   pylearn2_model = cPickle.load(f)

with open(words_path) as f:
   words = cPickle.load(f)

with open(chars_path) as f:
   char_dict = cPickle.load(f)
inv_dict = {v:k for k,v in char_dict.items()}
inv_dict[0] = inv_dict[len(inv_dict.keys())-1]
unknown =  inv_dict[0]

embeddings = np.load(embeddings_path)
print "embeddings", len(embeddings), embeddings[0].shape
# with open(embeddings_path) as f:
#    embeddings_path

print "Building Model"
# Change this to WordModel or CharModel depending on whther you are using word or character -based embeddings
fpropNoProjLayer = pylearn2_model.layers[0].fprop
fpropProjLayer = lambda state_below: pylearn2_model.layers[1].fprop(pylearn2_model.layers[0].fprop(state_below))
model = CharModel(pylearn2_model, char_dict, embeddings=embeddings, 
                  fprop=fpropNoProjLayer, words=words)
#model = WordModel(pylearn2_model, vocab, embeddings)

print "Calculating Closest Words"
if __name__ == "__main__":
    map(model.displayStringRun, ['cat', 'dog', 'France', 'france', 'Canada', 'Paris', 'paris', 'brother', 'mother',
                                 'sister', 'dad', 'mom', 'pharmacy', 'farm', 'quite', 'quiet', 'quit', 'like',
                                 'love', 'city', 'town'])
    #map(model.displayStringRun, ['monarch', 'democracy', 'political', 'raspberry', 'blueberry', 'accomplishment', 'applying', 'application'])
    map(model.displayIndexRun, range(100, 150))