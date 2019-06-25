# -*- coding:utf-8 -*-

import logging


from gensim.models.word2vec import LineSentence, Word2Vec


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# raw_sentences = ["the quick brown fox jumps over the lazy dogs","yoyoyo you go home now to sleep"]

sentences= LineSentence("../data/test/test.token .sbt")

model = Word2Vec(sentences ,min_count=1, iter=1000,size=150)
model.train(sentences, total_examples=model.corpus_count, epochs=1000)

model.save("../model/ast2v.mod")

model_loaded = Word2Vec.load("../model/ast2v.mod")

# sim = model_loaded.wv.most_similar(positive=[u'comment'])
# for s in sim:
#     print (s[0])

