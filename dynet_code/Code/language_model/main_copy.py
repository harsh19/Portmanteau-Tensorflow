import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import numpy as np
import csv
import configuration as config
from sklearn.preprocessing import LabelEncoder
import models
import pickle
import utilities_lm as datasets
import utilities_lm
from keras.callbacks import ModelCheckpoint


class PreProcessing:

	def __init__(self):
		self.unknown_word = "UNKNOWN"
		self.sent_start = "SENT_START"
		self.sent_end = "SENT_END"
		
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
		texts = [self.sent_start + " " + text + " " + self.sent_end for text in texts]
		
		tokenizer = Tokenizer(nb_words=config.MAX_VOCAB_SIZE)
		tokenizer.fit_on_texts(texts)
		sequences = tokenizer.texts_to_sequences(texts)

		word_index = tokenizer.word_index
		print('Found %s unique tokens.' % len(word_index))

		sequences = pad_sequences(sequences, maxlen=config.MAX_SEQUENCE_LENGTH)
		#sequences = [ np_utils.to_categorical(sequence) for sequence in sequences]
		
		a=sequences
		self.sequences = sequences
		print self.sequences[0]
		word_index[self.unknown_word]=0
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
		print self.x_train.shape
		print self.x_val.shape
		print self.x_test.shape
	
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
	rnn_model = models.RNNModel()

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
	params['inp_length'] = config.inp_length-1
	model = rnn_model.getModel(params)
	
	x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.x_train, preprocessing.y_train, preprocessing.x_val, preprocessing.y_val, preprocessing.x_test, preprocessing.y_test
	# train
        checkpointer = ModelCheckpoint(filepath="./checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1, save_best_only=True)
        model.fit(x_train, y_train, validation_data=(x_val, y_val),
              nb_epoch=config.num_epochs, batch_size=config.batch_size, callbacks=[checkpointer]) #config.num_epochs
	saveEmbeddings(model, preprocessing.word_index)
              
	#evaluate
	scores = model.evaluate(x_test, y_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))

	print "--- Sampling few sequences.. "
	for i in range(5):
	    pred = utilities.generateSentence(model, preprocessing.word_index, preprocessing.sent_start, 
	    	preprocessing.sent_end, preprocessing.unknown_word)
	    sent = [preprocessing.index_word[i] for i in pred]
	    if config.char_or_word==config.character_model:
	    	print ''.join(sent)
	    else:
	    	print ' '.join(sent)

if __name__ == "__main__":
	main()
