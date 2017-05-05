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
				if padding=='post':
					sequence = sequence + [0]*(maxlen - len(sequence))
				else:
					sequence = [0]*(maxlen - len(sequence)) + sequence
			ret.append(sequence)
		return np.array(ret)

	def loadPortmanteauData(self, src):
		data = open(src,"r").readlines()
		inputs, outputs = [],[]
		data = [row.strip().split(',') for row in data]
		inputs = [row[0]+" "+row[1] for row in data] # each input if firstWord SPACE secondWord
		outputs = [row[2] for row in data]
		
		char_to_idx = {}
		char_to_idx_ctr = 0 
		idx_to_char = {}

		char_to_idx[self.pad_word] = char_to_idx_ctr # 0 is for padword
		idx_to_char[char_to_idx_ctr]=self.pad_word
		char_to_idx_ctr+=1
		char_to_idx[self.sent_start] = char_to_idx_ctr
		idx_to_char[char_to_idx_ctr]=self.sent_start
		char_to_idx_ctr+=1
		char_to_idx[self.sent_end] = char_to_idx_ctr
		idx_to_char[char_to_idx_ctr]=self.sent_end		
		char_to_idx_ctr+=1

		texts = inputs
		for text in texts:
			for ch in text:
				if ch not in char_to_idx:
					char_to_idx[ch] = char_to_idx_ctr
					idx_to_char[char_to_idx_ctr]=ch
					char_to_idx_ctr+=1
		texts = outputs
		for text in texts:
			for ch in text:
				if ch not in char_to_idx:
					char_to_idx[ch] = char_to_idx_ctr
					idx_to_char[char_to_idx_ctr]=ch
					char_to_idx_ctr+=1

		print "Ignoring MAX_VOCAB_SIZE "
		print "Found vocab size = ",char_to_idx_ctr-1
		sequences_input = [ [char_to_idx[ch] for ch in text] for text in inputs ]
		sequences_input = [ [char_to_idx[self.sent_start]]+text+[char_to_idx[self.sent_end]] for text in sequences_input ]
		sequences_output = [ [char_to_idx[ch] for ch in text] for text in outputs ]
		sequences_output = [ [char_to_idx[self.sent_start]]+text+[char_to_idx[self.sent_end]] for text in sequences_output ]

		sequences_input = pad_sequences(sequences_input, maxlen=config.max_input_seq_length, padding='pre', truncating='post')
		sequences_output = pad_sequences(sequences_output, maxlen=config.max_output_seq_length, padding='post', truncating='post')
		
		self.word_index = char_to_idx
		self.index_word = idx_to_char
		self.vocab_size = len(char_to_idx)

		print "Printing few sample sequences... "
		print sequences_input[0],":", self.fromIdxSeqToVocabSeq(sequences_input[0]), "---", sequences_output[0], ":", self.fromIdxSeqToVocabSeq(sequences_output[0])
		print sequences_input[113], sequences_output[113]

		return sequences_input, sequences_output

	def fromIdxSeqToVocabSeq(self, seq):
		return [self.index_word[x] for x in seq]

	def prepareMTData(self, inputs, outputs, seed=123):

		decoder_inputs = np.array( [ sequence[:-1] for sequence in outputs ] )
		decoder_outputs = np.array( [ np.expand_dims(sequence[1:],-1) for sequence in outputs ] )
		encoder_inputs = np.array(inputs)

		indices = np.arange(encoder_inputs.shape[0])
		
		#shuffling
		np.random.seed(seed)
		np.random.shuffle(indices)
		
		#spplit indices
		nb_validation_samples = int(config.VALIDATION_SPLIT * encoder_inputs.shape[0])
		nb_test_samples = int(config.TEST_SPLIT * encoder_inputs.shape[0])
		test_indices = indices[-nb_test_samples:]
		val_indices = indices[-nb_test_samples-nb_validation_samples:nb_test_samples]
		train_indices = indices[0:-nb_test_samples-nb_validation_samples]
		print "nb_test_samples=",nb_test_samples

		#splits
		data = [encoder_inputs, decoder_inputs, decoder_outputs]
		train = [ dat[train_indices] for dat in data ]
		val = [ dat[val_indices] for dat in data ]
		test = [ dat[test_indices] for dat in data ]

		print "========================="
		print "traindata  lengths"
		for dat in train:
			print len(dat)
		print "========================="

		return train,val,test
		

	
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

	inputs,outputs = preprocessing.loadPortmanteauData("./data/finalports.csv")		
	train,val,test = preprocessing.prepareMTData(inputs,outputs)
	#return
	
	# get model
	params = {}
	params['embeddings_dim'] =  config.embeddings_dim
	params['lstm_cell_size'] = config.lstm_cell_size
	params['vocab_size'] =  preprocessing.vocab_size
	params['max_input_seq_length'] = config.max_input_seq_length
	params['max_output_seq_length'] = config.max_output_seq_length-1 #inputs are all but last element, outputs are al but first element
	params['batch_size'] = 20
	params['pretrained_embeddings']=True
	
	#return
	print params
	buckets = {  0:{'max_input_seq_length':40, 'max_output_seq_length':19},1:{'max_input_seq_length':40,'max_output_seq_length':19}, 2:{'max_input_seq_length':40, 'max_output_seq_length':19} }
	print buckets
	
	# train
	lim=200
	if lim!=-1:
		train_encoder_inputs, train_decoder_inputs, train_decoder_outputs = train
		train_encoder_inputs = train_encoder_inputs[:lim]
		train_decoder_inputs = train_decoder_inputs[:lim]
		train_decoder_outputs = train_decoder_outputs[:lim]
		train = train_encoder_inputs, train_decoder_inputs, train_decoder_outputs
	if params['pretrained_embeddings']:
		encoder_embedding_matrix = np.random.rand( params['vocab_size'], params['embeddings_dim'] )

	train_buckets = {}
	for bucket,_ in enumerate(buckets):
		train_buckets[bucket] = train

	#rnn_model = solver.Solver(buckets)
	#_ = rnn_model.getModel(params, mode='train',reuse=False, buckets=buckets)
	#rnn_model.trainModel(config=params, train_feed_dict=train_buckets, val_feed_dct=None, reverse_vocab=preprocessing.index_word, do_init=True)

	if len(train_decoder_outputs.shape)==3:
		train_decoder_outputs=np.reshape(train_decoder_outputs, (train_decoder_outputs.shape[0], train_decoder_outputs.shape[1]))

	rnn_model = solver.Solver(buckets=None, mode='inference')
	_ = rnn_model.getModel(params, mode='inference', reuse=False, buckets=None)
	print "----Running inference-----"
	rnn_model.runInference(params, train_encoder_inputs[:params['batch_size']], train_decoder_outputs[:params['batch_size']], preprocessing.index_word)

if __name__ == "__main__":
	main()
