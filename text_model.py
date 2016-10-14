#coding:utf-8
import numpy as np
from gensim.models import word2vec
from gensim.models import doc2vec

# ---------------------
# gensim word2vec
# ---------------------
# word 2 vectoe model building
def gw2v_model_train( corpus, model_size ):
	print 'load corpus ...'
	sentences = word2vec.Text8Corpus(corpus)
	print 'text2vec model generating ...'
	model = word2vec.Word2Vec(sentences, size=model_size, window=5, min_count=5, workers=4)
	print 'text2vec model saving ...'
	model.save('gw2v_sen2vec.model')

# sentence - unicode
def gw2v_sen2vec( sentence, word2vec_model, model_size ):
	word_list = sentence.split()
	sen_vec = np.zeros(shape=(1, model_size))
	for word in word_list:
		try:
			word_vec = word2vec_model[word]
		except KeyError:
			word_vec = np.zeros(shape=(1, model_size))
		sen_vec = sen_vec + word_vec
	sen_vec = sen_vec / len(word_list)
	sen_vec = np.nan_to_num(sen_vec)
	return sen_vec

# ---------------------
# gensim doc2vec
# ---------------------
def gd2v_model_train( corpus, model_size):
	print 'load corpus ...'
	documents = doc2vec.TaggedLineDocument(corpus)
	print 'text2vec model generating ...'
	model = doc2vec.Doc2Vec(documents, size=model_size, window=8, min_count=5, workers=4)
	print 'text2vec model saving ...'
	model.save('gd2v_sen2vec.model')

def gd2v_sen2vec( sentence, model, model_size ):
	word_list = sentence.split()
	sen_vec = model.infer_vector(word_list)
	return sen_vec
