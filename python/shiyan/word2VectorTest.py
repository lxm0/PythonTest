from gensim.models.word2vec import  Word2Vec

model_loaded = Word2Vec.load("model/ast2v.mod")

# sim = model_loaded.wv.most_similar(positive=[u'copy'])
sim = model_loaded.wv.similar_by_word('BasicType')
for s in sim:
    print (s[0],s[1])
vector = model_loaded.wv['BasicType']
print(vector)
print (model_loaded.wv.similarity('BasicType', 'FormalParameter'))