import numpy
import pickle

emb = numpy.load("embedding-dbpedia.npy").sum(-1)
mask = (emb == 0).nonzero()[0]

word_vocab = pickle.load(open("word_dict_cutoff4000.pickle"))
ivocab = {y:x for x,y in word_vocab.iteritems()}

print [ivocab[_] for _ in mask]