#coding:utf-8
import numpy as np
from gensim.models import word2vec

# word 2 vectoe model building
def word2vec_train( corpus, model_size ):
	sentences = word2vec.Text8Corpus(corpus)
	word2vec_model = word2vec.Word2Vec(sentences, size=model_size, window=5, min_count=5, workers=4)
	word2vec_model.save('word_vec.model')

# word 2 vec
def wor2vec_test( word2vec_model ):
	word = 'æ'
	print word
	word = word.decode('utf8')
	print word2vec_model[word]
	
	word = 'èèµ'
	print word
	word = word.decode('utf8')
	y2 = word2vec_model.most_similar(word, topn=20)
	for item in y2:
		print item[0].encode('utf8'), item[1]

# sentence 2 vector
# sentence - unicode
def sen2vec( sentence, word2vec_model, model_size ):
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
