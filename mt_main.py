from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import numpy as np
import csv
import configuration as config
from sklearn.preprocessing import LabelEncoder
import mt_model as models
import pickle
import utilities as datasets
import utilities
import mt_solver as solver

class PreProcessing:

	def __init__(self):
		self.unknown_word = "UNKNOWN"
		self.sent_start = "SENTSTART".lower()
		self.sent_end = "SENTEND".lower()
		self.pad_word = "PADWORD".lower()

	def pad_sequences_my(sequences, maxlen, padding='post', truncating='post'):
		ret=[]
		for sequence in sequences:
			if len(sequence)>=maxlen:
				sequence=sequence[:maxlen]
			else:
				sequence = sequence + [0]*(maxlen - len(sequence))
			ret.append(sequence)
		return np.array(ret)
		
	def loadDataCharacter(self, data=None):
		if data==None:
			print "loading data..."
			data_src = config.data_src
			texts = open(data_src,"r").readlines()
		else:
			texts = data
		char_to_idx = {}
		char_to_idx_ctr = 1
		idx_to_char = {}

		char_to_idx[self.sent_start] = char_to_idx_ctr
		idx_to_char[char_to_idx_ctr]=self.sent_start
		char_to_idx_ctr+=1
		char_to_idx[self.sent_end] = char_to_idx_ctr
		idx_to_char[char_to_idx_ctr]=self.sent_end		
		char_to_idx_ctr+=1
		for text in texts:
			for ch in text:
				if ch not in char_to_idx:
					char_to_idx[ch] = char_to_idx_ctr
					idx_to_char[char_to_idx_ctr]=ch
					char_to_idx_ctr+=1

		print "Ignoring MAX_VOCAB_SIZE "
		print "Found vocab size = ",char_to_idx_ctr-1
		sequences = [ [char_to_idx[ch] for ch in text] for text in texts ]
		sequences = [ [char_to_idx[self.sent_start]]+text+[char_to_idx[self.sent_end]] for text in sequences ]

		sequences = pad_sequences(sequences, maxlen=config.MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
		print "Printing few sample sequences... "
		print sequences[0]
		print sequences[113]
		print sequences[222]
		
		self.sequences = sequences
		char_to_idx[self.unknown_word]=0
		self.word_index = char_to_idx
		idx_to_char[0]=self.unknown_word
		self.index_word = idx_to_char
		self.vocab_size = len(char_to_idx) + 1 # for padded

	def loadData(self):   
		print "loading data..."
		data_src = config.data_src
		texts = open(data_src,"r").readlines()
		texts = [ text.strip() for text in texts ]
		texts = [self.sent_start + " " + text + " " + self.sent_end for text in texts]
		#print texts[0]
		
		tokenizer = Tokenizer(nb_words=config.MAX_VOCAB_SIZE)
		tokenizer.fit_on_texts(texts)
		sequences = tokenizer.texts_to_sequences(texts)


		word_index = tokenizer.word_index
		i_to_w = { i:w for w,i in word_index.items() }
		print('Found %s unique tokens.' % len(word_index))
		print [i_to_w[i] for i in sequences[0]]
		print ""

		sequences = pad_sequences(sequences, maxlen=config.MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
		#sequences = [ np_utils.to_categorical(sequence) for sequence in sequences]
		print sequences[0]
		print texts[0]

		print "*************** ", word_index[self.sent_start]

		a=sequences
		self.sequences = sequences
		print self.sequences[0]
		word_index[self.unknown_word]=len(word_index)
		word_index[self.pad_word]=0
		self.word_index = word_index
		index_word = {i:w for w,i in word_index.items()}
		self.index_word = index_word
		#print word_index

	def prepareLMdata(self,seed=123):

		data = np.array( [ sequence[:-1] for sequence in self.sequences ] )
		labels = np.array( [ np.expand_dims(sequence[1:],-1) for sequence in self.sequences ] )
		indices = np.arange(data.shape[0])
		np.random.seed(seed)
		np.random.shuffle(indices)
		data = data[indices]
		labels = labels[indices]
		nb_validation_samples = int(config.VALIDATION_SPLIT * data.shape[0])
		nb_test_samples = int(config.TEST_SPLIT * data.shape[0])
		print "nb_test_samples=",nb_test_samples

		self.x_train = data[0:-nb_test_samples-nb_validation_samples]
		self.y_train = labels[0:-nb_test_samples-nb_validation_samples]
		self.x_val = data[-nb_test_samples-nb_validation_samples:-nb_test_samples]
		self.y_val = labels[-nb_test_samples-nb_validation_samples:-nb_test_samples]
		self.x_test = data[-nb_test_samples:]
		self.y_test = labels[-nb_test_samples:]
		print "================="
		print self.x_train.shape, " ", self.y_train.shape
		print self.x_val.shape
		print self.x_test.shape
		print "================="

	
def saveEmbeddings(model, vocab, embeddings_out_name = "output_embeddings.txt"):
	layer = model.layers[1]
	print type(layer)
	wt = layer.get_weights()
	print type(wt)
	print len(wt)
	print type(wt[0])
	embeddings = wt[0]
	print embeddings.shape
	fw = open(embeddings_out_name, "w")
	for word,idx in vocab.items():
		fw.write(word + "\t")
		for val in embeddings[idx]:
			fw.write( str(val) + "\t")
		fw.write("\n")
	fw.close()
	print "Saved embeddings to ",embeddings_out_name


def main():
	preprocessing = PreProcessing()
	rnn_model = solver.Solver()

	if config.char_or_word == config.character_model:
		data=None
		if config.data_type=="cmu_dict":
			cmu_data = datasets.getCMUDictData(config.data_src_cmu)
			data=cmu_data
		preprocessing.loadDataCharacter(data=data)
	else:
		preprocessing.loadData()		
	preprocessing.prepareLMdata()
	
	# get model
	params = {}
	params['embeddings_dim'] =  config.embeddings_dim
	params['lstm_cell_size'] = config.lstm_cell_size
	if config.char_or_word == config.character_model:
		params['vocab_size'] =  preprocessing.vocab_size
	else:
		params['vocab_size'] =  len( preprocessing.word_index )
	params['max_sentence_length'] = config.inp_length-1
	params['batch_size'] = 20
	
	x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.x_train, preprocessing.y_train, preprocessing.x_val, preprocessing.y_val, preprocessing.x_test, preprocessing.y_test
	
	# train
	x_train = x_train[:200]
	y_train = y_train[:200]


	#_ = rnn_model.getModel(params, mode='train')
	#rnn_model.trainModel(x_train, y_train, params, None, None, preprocessing.index_word)
	
	_ = rnn_model.getModel(params, mode='inference')
	rnn_model.runInference(params, x_train[:params['batch_size']], preprocessing.index_word)

if __name__ == "__main__":
	main()
