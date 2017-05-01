import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Merge
from keras.utils import np_utils
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Input, Embedding, LSTM, Dense, merge, SimpleRNN, TimeDistributed
import tensorflow as tf
from tensorflow.contrib import rnn
from utilities import OutputSentence, TopN
import utilities
import mt_model as model
	
class Solver:

	def __init__(self):
		self.model_obj = model.RNNModel()


	def getModel(self, config, mode='train' ):

		if mode=='train':
			encoder_outputs = self.model_obj.getEncoderModel(config, mode='training', reuse=False)
			self.pred = self.model_obj.getDecoderModel(config, encoder_outputs, is_training=True, mode='training')
			#tf.get_variable_scope().reuse_variables()
			encoder_outputs = self.model_obj.getEncoderModel(config, mode='inference', reuse=True)
			print "encoder_outputs.shaoe :::: ",len(encoder_outputs),encoder_outputs[0].shape
			self.decoder_outputs_inference, self.encoder_outputs = self.model_obj.getDecoderModel(config, encoder_outputs, is_training=False, 	mode='inference', reuse=True)	
			print "elf.encoder_outputs.shaoe :::: ",len(encoder_outputs),self.encoder_outputs[0].shape
		else:
			config['batch_size'] = 5
			encoder_outputs = self.model_obj.getEncoderModel(config, mode='inference')
			print "encoder_outputs.shaoe :::: ",len(encoder_outputs),encoder_outputs[0].shape
			self.decoder_outputs_inference, self.encoder_outputs = self.model_obj.getDecoderModel(config, encoder_outputs, is_training=False, 	mode='inference', reuse=False)	
			print "elf.encoder_outputs.shaoe :::: ",len(encoder_outputs),self.encoder_outputs[0].shape
		print("============== \n Printing all trainainble variables")
		for v in tf.trainable_variables():
			print(v)
		print("==================")


	def trainModel(self, x_train, y_train, config, inference_config, val_feed_dct, reverse_vocab):
		
		print("============== \n Printing all trainainble variables")
		for v in tf.trainable_variables():
			print(v)
		print("==================")

		cost = self.model_obj.cost

		# if y is passed as (N, seq_length, 1): change it to (N,seq_length)
		if len(y_train.shape)==3:
			y_train=np.reshape(y_train, (y_train.shape[0], y_train.shape[1]))

		#create temporary feed dictionary
		feed_dct={self.model_obj.token_lookup_sequences_placeholder:x_train, self.model_obj.token_output_sequences_decoder_placeholder:y_train, self.model_obj.token_lookup_sequences_decoder_placeholder:x_train}

		# Gradient descent
		learning_rate=0.1
		batch_size=config['batch_size']
		optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

		# Initializing the variables
		init = tf.global_variables_initializer()

		training_iters=10
		display_step=2
		sample_step=2
		n = feed_dct[self.model_obj.token_lookup_sequences_placeholder].shape[0]
		# Launch the graph
		with tf.Session() as sess:
			sess.run(init)
			step = 1
			#preds = np.array( sess.run(self.pred, feed_dict= feed_dct) )
			#print preds
			while step < training_iters:
				#num_of_batches =  n/batch_size #(n+batch_size-1)/batch_size
				num_of_batches =  (n+batch_size-1)/batch_size
				for j in range(num_of_batches):
					feed_dict_cur = {}
					for k,v in feed_dct.items():
						#feed_dict_cur[k] = v[j*batch_size:(j+1)*batch_size]
						feed_dict_cur[k] = v[j*batch_size:min(n,(j+1)*batch_size)]
						#print k
						#print feed_dict_cur[k].shape
					cur_out = feed_dict_cur[self.model_obj.token_output_sequences_decoder_placeholder]
					x,y = np.nonzero(cur_out)
					mask = np.zeros(cur_out.shape, dtype=np.float)
					mask[x,y]=1
					#print mask[0]
					#print cur_out[0]
					feed_dict_cur[self.model_obj.masker]=mask
					#feed_dict_cur[self.state_placeholder] = np.zeros((2,config['batch_size'],config['lstm_cell_size']))

					sess.run(optimizer, feed_dict=feed_dict_cur )
					if step % display_step == 0:
						if j<10:
							loss = sess.run(cost, feed_dict= feed_dict_cur)
							print "step ",step," : ",loss
					if step % sample_step == 0:
						#continue
						if j==0:
		  					self.runInference( config, x_train[:batch_size], reverse_vocab, sess )
							pred = np.array( sess.run(self.pred, feed_dict= feed_dict_cur) )
							print pred.shape
							print pred[0].shape
							print np.sum(pred[0],axis=1)
							


				step += 1
			saver = tf.train.Saver()
			save_path = saver.save(sess, "/tmp/model.ckpt")
  			print "Model saved in file: ",save_path
			self.saver = saver

	###################################################################################

	def runInference(self, config, x_test, reverse_vocab, sess=None): # sampling
		#print "============================================================"
		if sess==None:
	  		sess = tf.Session()
	  		saver = tf.train.Saver()
	  		saver.restore(sess, "/tmp/model.ckpt")
		#return utilities.runInference(config, x_test, reverse_vocab, sess, solver_obj =self)
		typ = "greedy" #config['inference_type']
		model_obj = self.model_obj
		#feed_dct={model_obj.token_lookup_sequences_placeholder:x_test}
		feed_dct={model_obj.token_lookup_sequences_placeholder_inference:x_test}
		batch_size = config['batch_size'] #x_test.shape[0]
		max_sentence_length = config['max_sentence_length']
		if typ=="greedy":
			decoder_outputs_inference, encoder_outputs = np.array( sess.run([self.decoder_outputs_inference, self.encoder_outputs], feed_dict= feed_dct) ) # timesteps, N
			print("----->>>>>>>>")
			print(encoder_outputs.shape)
			print(encoder_outputs[0][1][:15]) # 1st data point, 2nd word
			print(encoder_outputs[1][1][:15]) # 2nd data point, 2nd word
			print("----->>>>>>>>")
			decoder_outputs_inference = np.transpose(decoder_outputs_inference) # (N,timesteps)
			#print "decoder_outputs_inference.shape : ",decoder_outputs_inference.shape
			for i,row in enumerate(decoder_outputs_inference):
				ret=""
				for val in row:
					ret+=( " " + reverse_vocab[val] )
				print "GT: ", [ reverse_vocab[j] for j in x_test[i]]
				print "prediction: ",ret
				print "row= ",row
				print "matches: ", [ r==x for r,x in zip(row,x_test[i]) ]
				print ""
				if i>20:
					break

########################################################################################
