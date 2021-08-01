
import numpy as np
import pathlib
import pickle

#from tensorflow import cast, concat, constant
#from tensorflow import zeros, ones, shape, reshape, expand_dims, reduce_sum
#from tensorflow import GradientTape, TensorSpec, Module
#from tensorflow import range as tf_range
#from tensorflow import string as tf_string
#from tensorflow import function as tf_function
#from tensorflow import strings as tf_strings
#from tensorflow.math import tanh

import tensorflow as tf

# Change name for deep packaging 
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Loss, SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import Callback

from tensorflow.keras.layers import Layer, Input, Embedding, Dense
from tensorflow.keras.layers import GRU, LSTM, Bidirectional, AdditiveAttention
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization, StringLookup

from auxiliar_functions import english_standardization, spanish_standardization, french_standardization

# additive attention: Bahdanau
# Multiplicative attention: Luong

# super(): referes to the base class

# Explain of some common tensorflow basis methods:
# tf.cast: change tensor type
# tf.reduce_sum: sum array elements along the dimension specified by axis



# ===============================================
#   M O D E L   C O M P O N E N T S 
# ===============================================


class EncoderGRU(Layer):

	def __init__(self, input_vocab_size, embedding_dim, enc_units=128):
		super(EncoderGRU, self).__init__()
		self.embedding = Embedding(input_vocab_size, embedding_dim)
		self.gru = GRU(enc_units, return_sequences=True,
		               return_state=True, recurrent_initializer='glorot_uniform')

	def call(self, tokens, state=None):
		vectors = self.embedding(tokens)
		output, state = self.gru(vectors, initial_state=state)
		return output, state

class EncoderLSTM(Layer):

	def __init__(self, input_vocab_size, embedding_dim, enc_units=128):
		super(EncoderLSTM, self).__init__()
		self.embedding = Embedding(input_vocab_size, embedding_dim)
		self.lstm = LSTM(enc_units, return_sequences=True,
		               return_state=True, recurrent_initializer='glorot_uniform')

	def call(self, tokens, state=None):
		vectors = self.embedding(tokens)
		output, state_h, state_c = self.lstm(vectors, initial_state=state)
		return output, [state_h, state_c]

# still to test: how are the outputs of a bidirectional layer organized? and the states?
class EncoderBiLSTM(Layer):

	def __init__(self, input_vocab_size, embedding_dim, enc_units=128):
		super(EncoderBiLSTM, self).__init__()
		self.embedding = Embedding(input_vocab_size, embedding_dim)
		self.bilstm = Bidirectional(LSTM(enc_units, return_sequences=True,
		               return_state=True, recurrent_initializer='glorot_uniform'))

	def call(self, tokens, state=None):
		vectors = self.embedding(tokens)
		output, state_h, state_c = self.bilstm(vectors, initial_state=state)
		return output, [state_h, state_c]

# # query: states of decoder
# 	# value: encoder output
# 	# then, context vector is concatenated to embedding output
# class BahdanauAttention(Layer):
	
# 	def __init__(self, units):
# 		super().__init__()
# 		self.W1 = Dense(units, use_bias=False) # lin comb query (decoder state)
# 		self.W2 = Dense(units, use_bias=False) # lin comb value (encoder output)
# 		self.attention = AdditiveAttention() # inner weight vector of length units

# 	def call(self, query, value):
# 		w1_query = self.W1(query)
# 		w2_key = self.W2(value)
# 		#query_mask = ones( shape(query)[:-1], dtype=bool )
# 		#value_mask = mask
# 		context_vector, attention_weights = self.attention(
# 		    inputs = [w1_query, value, w2_key],
# 		    #mask=[query_mask, value_mask],
# 		    return_attention_scores = True )

# 		return context_vector, attention_weights


class BahdanauAttention(Layer):
	
	def __init__(self, units):
		super().__init__()
		self.W1 = Dense(units, use_bias=False) # lin comb query (decoder state)
		self.W2 = Dense(units, use_bias=False) # lin comb value (encoder output)
		self.attention = AdditiveAttention()
	
	def call(self, query, value):
		w1_query = self.W1(query)
		w2_key = self.W2(value)
		context_vector, attention_weights = self.attention(
		    inputs = [w1_query, value, w2_key],
		    return_attention_scores = True )

		return context_vector, attention_weights



class DecoderAttGRU(Layer):
	def __init__(self, output_vocab_size, embedding_dim, dec_units):
		super(DecoderAttGRU, self).__init__()
		self.embedding = Embedding(output_vocab_size, embedding_dim)
		self.gru = GRU(dec_units, return_sequences=True,
		                return_state=True, recurrent_initializer='glorot_uniform')
		self.attention = BahdanauAttention(dec_units)

		# receives attention and decoder GRU output 
		self.Wc = Dense(dec_units, activation=tf.tanh, use_bias=False)
		self.fc = Dense(output_vocab_size)

	def call(self, new_tokens, enc_output, state=None):

		# process decoder tokens
		vectors = self.embedding(new_tokens)
		rnn_output, state = self.gru(vectors, initial_state=state)

		# attention mechanism
		# process encoder outputs and decoder RNN states
		context_vector, attention_weights = self.attention(rnn_output, enc_output)

		# process attention and RNN output jointly
		context_and_rnn_output = tf.concat([context_vector, rnn_output], axis=-1)
		attention_vector = self.Wc(context_and_rnn_output)
		logits = self.fc(attention_vector)

		return logits, state, attention_weights


class DecoderAttLSTM(Layer):
	def __init__(self, output_vocab_size, embedding_dim, dec_units):
		super(DecoderAttLSTM, self).__init__()
		self.embedding = Embedding(output_vocab_size, embedding_dim)
		self.lstm = LSTM(dec_units, return_sequences=True,
		                return_state=True, recurrent_initializer='glorot_uniform')
		self.attention = BahdanauAttention(dec_units)

		# receives attention and decoder GRU output 
		self.Wc = Dense(dec_units, activation=tf.tanh, use_bias=False)
		self.fc = Dense(output_vocab_size)

	def call(self, new_tokens, enc_output, state=None):

		# process decoder tokens
		vectors = self.embedding(new_tokens)
		rnn_output, state_h, state_c = self.lstm(vectors, initial_state=state)

		# attention mechanism
		# process encoder outputs and decoder RNN states
		context_vector, attention_weights = self.attention(rnn_output, enc_output)

		# process attention and RNN output jointly
		context_and_rnn_output = tf.concat([context_vector, rnn_output], axis=-1)
		attention_vector = self.Wc(context_and_rnn_output)
		logits = self.fc(attention_vector)

		return logits, [state_h, state_c], attention_weights


class DecoderGRU(Layer):
	def __init__(self, output_vocab_size, embedding_dim, dec_units):
		super(DecoderGRU, self).__init__()
		self.embedding = Embedding(output_vocab_size, embedding_dim)
		self.gru = GRU(dec_units, return_sequences=True,
		                return_state=True, recurrent_initializer='glorot_uniform')
		self.Wc = Dense(dec_units, activation=tf.tanh)
		self.fc = Dense(output_vocab_size)

	def call(self, new_tokens, state=None):
		# process decoder tokens
		vectors = self.embedding(new_tokens)
		x, state = self.gru(vectors, initial_state=state)
		x = self.Wc(x)
		logits = self.fc(x)

		return logits, state


def load_encoder_vectorizer(max_vocab_enc, vocab_file_enc):
	path_vocab_enc = pathlib.Path(vocab_file_enc)
	text = path_vocab_enc.read_text(encoding='utf-8')
	lines = text.splitlines()
	input_text_processor = TextVectorization(
		standardize=english_standardization,
		max_tokens=max_vocab_enc,
		vocabulary=lines)
	return input_text_processor

def load_decoder_vectorizer(max_vocab_dec, vocab_file_dec, language):
	path_vocab_dec = pathlib.Path(vocab_file_dec)
	text = path_vocab_dec.read_text(encoding='utf-8')
	lines = text.splitlines() # list of strings
	if language == 'portuguese' or 'spanish':
		output_text_processor = TextVectorization(
			standardize=spanish_standardization,
			max_tokens=max_vocab_dec,
			vocabulary=lines)
	elif language == 'french':
		output_text_processor = TextVectorization(
			standardize=french_standardization,
			max_tokens=max_vocab_dec,
			vocabulary=lines)

	return output_text_processor



# ===============================================
#   A U X I L I A R   C L A S S E S
# ===============================================

# Function to compute loss at every step
#  - uses categorical cross-entropy (integer label)
#  - mask to exclude where labels are zero (out of range)
class MaskedLoss(Loss):
	def __init__(self):
		self.name = 'masked_loss'
		self.loss = SparseCategoricalCrossentropy(from_logits=True, reduction='none')

	def __call__(self, y_true, y_pred):
		# Calculate the loss for each item in the batch.
		_loss = self.loss(y_true, y_pred)

		# Mask off the losses on padding.
		mask = tf.cast(y_true != 0, dtype=_loss.dtype)
		_loss *= mask

		# Return the total.
		return tf.reduce_sum(_loss)


# Callback objects access the caller model as attribute (self.model)
# Save model parameters, overwriting the former one
class CustomCheckpoint(Callback):
	def __init__(self, folderpath):
		self.best = np.inf
		self.folder = folderpath

	def on_epoch_end(self, epoch, logs=None):
		#keys = list(logs.keys())
		#print("End epoch {} of training; got log keys: {}".format(epoch, keys))
		val_loss = logs['val_loss']
		if val_loss <= self.best:
			self.best = val_loss
			self.model.export_parameters(self.folder)




# ===============================================
#   C O M P L E T E   M O D E L 
# ===============================================

# translator.build(Input(shape=(None), dtype=tf_string, name='text'))

class TrainTranslatorGRU(Model):

	def __init__(self, embedding_dim, units,
				max_vocab_enc, max_vocab_dec, vocabfile1, vocabfile2, language):
		super().__init__()

		# (1) Load preprocessors
		self.input_text_processor = load_encoder_vectorizer(max_vocab_enc,vocabfile1)
		self.output_text_processor = load_decoder_vectorizer(max_vocab_dec, vocabfile2, language)

		# (2) Build the encoder and decoder
		self.encoder = EncoderGRU(self.input_text_processor.vocabulary_size(), embedding_dim, units)
		self.decoder = DecoderGRU(self.output_text_processor.vocabulary_size(), embedding_dim, units)
		
		# (3) build (set input size to initialize weights)
		enc_text = Input(shape=(1,), dtype=tf.string, name='enc')
		x = self.input_text_processor(enc_text)
		_, s = self.encoder(x)

		dec_text = Input(shape=(1,), dtype=tf.string, name='dec')
		x = self.output_text_processor(dec_text)
		x, _  = self.decoder(x, state = s )


    # INTERNAL FUNCTIONS

	def _decoder_steps(self, new_tokens, dec_state):
		input_token, target_token = new_tokens[:, 0:1], new_tokens[:, 1:2]
		logits, dec_state = self.decoder(input_token, state=dec_state)
		step_loss = self.loss(target_token, logits)
		return step_loss, dec_state

	@tf.function( input_signature=[[ tf.TensorSpec(dtype=tf.string, shape=[None]),
		tf.TensorSpec(dtype=tf.string, shape=[None])]] )
	def train_step(self, inputs):

		input_text, target_text = inputs

		# Tokenize encoder and decoder sequences
		#(input_tokens, input_mask, target_tokens, target_mask) = self._preprocess(input_text, target_text)
		input_tokens = self.input_text_processor(input_text)
		target_tokens = self.output_text_processor(target_text)
		max_target_length = tf.shape(target_tokens)[1]

		with tf.GradientTape() as tape:

			# (1) Forward computation of encoder
			_, enc_state = self.encoder(input_tokens)

			# Initialize the decoder state
			# (this only works if the encoder and decoder have the same number of units.)
			dec_state = enc_state
			loss = tf.constant(0.0)

			# (2) Compute decoder output for each target sequence elements
			for t in tf.range(max_target_length-1):
				new_tokens = target_tokens[:, t:t+2] # two tokens: current and next for prediction
				step_loss, dec_state = self._decoder_steps(new_tokens, dec_state)
				loss += step_loss
			# Average the loss over all non padding tokens.
			average_loss = loss / tf.reduce_sum(tf.cast(target_tokens!=0, loss.dtype))

		# Apply an optimization step
		variables = self.trainable_variables 
		gradients = tape.gradient(average_loss, variables)
		self.optimizer.apply_gradients(zip(gradients, variables))

		# metrics: update state (would have to output y_pred and compare to target_token)
		#self.compiled_metrics.update_state(y, y_pred)

		# Return a dict mapping metric names to current value
		return {'loss': average_loss}


	@tf.function( input_signature=[[ tf.TensorSpec(dtype=tf.string, shape=[None]),
		tf.TensorSpec(dtype=tf.string, shape=[None])]] )
	def test_step(self, inputs):

		# (1) preprocessing
		input_text, target_text = inputs
		input_tokens = self.input_text_processor(input_text)
		target_tokens = self.output_text_processor(target_text)
		max_target_length = tf.shape(target_tokens)[1]

		# (2) encoding
		_, enc_state = self.encoder(input_tokens)
		dec_state = enc_state
		loss = tf.constant(0.0)

		# (3) decoding
		for t in tf.range(max_target_length-1):
			new_tokens = target_tokens[:, t:t+2] # two tokens: current and next for prediction
			step_loss, dec_state = self._decoder_steps(new_tokens, dec_state)
			loss += step_loss
		# Average the loss over all non padding tokens.
		average_loss = loss / tf.reduce_sum(tf.cast(target_tokens!=0, loss.dtype))

		# update metric state
		# val_acc_metric.update_state(y, val_logits)
		return {'loss': average_loss }
			

	def export_parameters(self, folderpath):
		enc_weights = self.encoder.get_weights()
		dec_weights = self.decoder.get_weights()

		with open(folderpath+'/weights_encoder.pickle', 'wb') as handle:
			pickle.dump(enc_weights, handle)

		with open(folderpath+'/weights_decoder.pickle', 'wb') as handle:
			pickle.dump(dec_weights, handle)

	def import_parameters(self, folderpath):

		with open(folderpath+'/weights_encoder.pickle', 'rb') as handle:
			enc_weights = pickle.load(handle)
		with open(folderpath+'/weights_decoder.pickle', 'rb') as handle:
			dec_weights = pickle.load(handle)
		self.encoder.set_weights(enc_weights)
		self.decoder.set_weights(dec_weights)



class TrainTranslatorAttGRU(Model):

	def __init__(self, embedding_dim, units,
				max_vocab_enc, max_vocab_dec, vocabfile1, vocabfile2, language):
		super().__init__()

		# (1) Load preprocessors
		self.input_text_processor = load_encoder_vectorizer(max_vocab_enc,vocabfile1)
		self.output_text_processor = load_decoder_vectorizer(max_vocab_dec, vocabfile2, language)

		# (2) Build the encoder and decoder
		self.encoder = EncoderGRU(self.input_text_processor.vocabulary_size(), embedding_dim, units)
		self.decoder = DecoderAttGRU(self.output_text_processor.vocabulary_size(), embedding_dim, units)
		
		# (3) build (set input size to initialize weights)
		enc_text = Input(shape=(1,), dtype=tf.string, name='enc')
		x = self.input_text_processor(enc_text)
		out1, s = self.encoder(x)

		dec_text = Input(shape=(1,), dtype=tf.string, name='dec')
		x = self.output_text_processor(dec_text)
		x, _, _  = self.decoder(x, out1, state = s )


    # INTERNAL FUNCTIONS

	def _decoder_steps(self, new_tokens, enc_output, dec_state):
		input_token, target_token = new_tokens[:, 0:1], new_tokens[:, 1:2]
		logits, dec_state, _ = self.decoder(input_token, enc_output, state=dec_state)

		step_loss = self.loss(target_token, logits)

		return step_loss, dec_state

	@tf.function( input_signature=[[ tf.TensorSpec(dtype=tf.string, shape=[None]),
		tf.TensorSpec(dtype=tf.string, shape=[None])]] )
	def train_step(self, inputs):

		input_text, target_text = inputs

		# Tokenize encoder and decoder sequences
		#(input_tokens, input_mask, target_tokens, target_mask) = self._preprocess(input_text, target_text)
		input_tokens = self.input_text_processor(input_text)
		target_tokens = self.output_text_processor(target_text)
		max_target_length = tf.shape(target_tokens)[1]

		with tf.GradientTape() as tape:

			# (1) Forward computation of encoder
			enc_output, enc_state = self.encoder(input_tokens)

			# Initialize the decoder state
			# (this only works if the encoder and decoder have the same number of units.)
			dec_state = enc_state
			loss = tf.constant(0.0)

			# (2) Compute decoder output for each target sequence elements
			for t in tf.range(max_target_length-1):
				new_tokens = target_tokens[:, t:t+2] # two tokens: current and next for prediction
				step_loss, dec_state = self._decoder_steps(new_tokens, enc_output, dec_state)
				loss += step_loss
			# Average the loss over all non padding tokens.
			average_loss = loss / tf.reduce_sum(tf.cast(target_tokens!=0, loss.dtype))

		# Apply an optimization step
		variables = self.trainable_variables 
		gradients = tape.gradient(average_loss, variables)
		self.optimizer.apply_gradients(zip(gradients, variables))

		# metrics: update state (would have to output y_pred and compare to target_token)
		#self.compiled_metrics.update_state(y, y_pred)

		# Return a dict mapping metric names to current value
		return {'loss': average_loss}


	@tf.function( input_signature=[[ tf.TensorSpec(dtype=tf.string, shape=[None]),
		tf.TensorSpec(dtype=tf.string, shape=[None])]] )
	def test_step(self, inputs):

		# (1) preprocessing
		input_text, target_text = inputs
		input_tokens = self.input_text_processor(input_text)
		target_tokens = self.output_text_processor(target_text)
		max_target_length = tf.shape(target_tokens)[1]

		# (2) encoding
		enc_output, enc_state = self.encoder(input_tokens)
		dec_state = enc_state
		loss = tf.constant(0.0)

		# (3) decoding
		for t in tf.range(max_target_length-1):
			new_tokens = target_tokens[:, t:t+2] # two tokens: current and next for prediction
			step_loss, dec_state = self._decoder_steps(new_tokens, enc_output, dec_state)
			loss += step_loss
		# Average the loss over all non padding tokens.
		average_loss = loss / tf.reduce_sum(tf.cast(target_tokens!=0, loss.dtype))

		# update metric state
		# val_acc_metric.update_state(y, val_logits)
		return {'loss': average_loss }
			

	def export_parameters(self, folderpath):
		enc_weights = self.encoder.get_weights()
		dec_weights = self.decoder.get_weights()

		with open(folderpath+'/weights_encoder.pickle', 'wb') as handle:
			pickle.dump(enc_weights, handle)

		with open(folderpath+'/weights_decoder.pickle', 'wb') as handle:
			pickle.dump(dec_weights, handle)

	def import_parameters(self, folderpath):

		with open(folderpath+'/weights_encoder.pickle', 'rb') as handle:
			enc_weights = pickle.load(handle)
		with open(folderpath+'/weights_decoder.pickle', 'rb') as handle:
			dec_weights = pickle.load(handle)
		self.encoder.set_weights(enc_weights)
		self.decoder.set_weights(dec_weights)




# ===============================================
#   P R E D I C T I O N   T R A N S L A T O R
# ===============================================


class TranslatorGRU(tf.Module):

	def __init__(self, max_vocab_enc, max_vocab_dec, vocabfile_enc, vocabfile_dec,
				language, embedding_dim, hidden_dim, results_folder):

		# (1) Load tokenizer (convert input sentence to tokens)
		self.input_text_processor = load_encoder_vectorizer(max_vocab_enc,vocabfile_enc)
		output_text_processor = load_decoder_vectorizer(max_vocab_dec,vocabfile_dec,language)

		# (2) Load de-tokenizer (convert predicted tokens to words)
		dec_vocabfile2 = pathlib.Path(vocabfile_dec)
		text = dec_vocabfile2.read_text(encoding='utf-8')
		vocab_dec = text.splitlines()
		self.output_detokenizer = StringLookup( vocabulary=vocab_dec, invert=True)

		# (3) Special tokens and prediction mask
		# (mask to prohibit '', [UNK] or [BOS] during prediction)
		output_tokenizer = StringLookup( vocabulary=vocab_dec )
		self.start_token = output_tokenizer('[BOS]')
		self.end_token = output_tokenizer('[EOS]')
		token_mask_ids = output_tokenizer(['', '[UNK]', '[BOS]']).numpy()
		token_mask = np.zeros([len(vocab_dec)], dtype=np.bool)
		token_mask[np.array(token_mask_ids)] = True
		self.token_mask = token_mask

		# (4) Load encoder and decoder, and load weights
		self.encoder = EncoderGRU(max_vocab_enc, embedding_dim, hidden_dim)
		self.decoder = DecoderGRU(max_vocab_dec, embedding_dim, hidden_dim)

		# build before load weights
		enc_text = Input(shape=(1,), dtype=tf.string, name='enc')
		x = self.input_text_processor(enc_text)
		_, s = self.encoder(x)
		dec_text = Input(shape=(1,), dtype=tf.string, name='dec')
		x = output_text_processor(dec_text)
		x, _  = self.decoder(x, state = s )

		self.import_parameters(results_folder)


	def import_parameters(self, folderpath):
		with open(folderpath+'/weights_encoder.pickle', 'rb') as handle:
			enc_weights = pickle.load(handle)
		with open(folderpath+'/weights_decoder.pickle', 'rb') as handle:
			dec_weights = pickle.load(handle)
		self.encoder.set_weights(enc_weights)
		self.decoder.set_weights(dec_weights)

	def tokens_to_text(self, tokens):
		words = self.output_detokenizer(tokens)
		sentence = tf.strings.reduce_join(words, axis=1, separator=' ')
		sentence = tf.strings.strip(sentence)
		return sentence

	def sample_token(self, logits, temperature):

		# masking: where mask=True, take logit = -np.inf
		token_mask = self.token_mask[tf.newaxis, tf.newaxis, :] # extend array dimensions
		logits = tf.where(self.token_mask, -np.inf, logits)

		if temperature == 0.0: # greedy sampling
			new_tokens = tf.argmax(logits, axis=-1)
		else: # beam sampling
			logits = tf.squeeze(logits, axis=1)
			new_tokens = tf.random.categorical(logits/temperature, num_samples=1)

		return new_tokens

	def translate(self, input_text, max_length=50, temperature=1.0):

		batch_size = tf.shape(input_text)[0]
		
		# encoder
		input_tokens = self.input_text_processor(input_text)
		_, enc_state = self.encoder(input_tokens)

		# decoding preparation
		dec_state = enc_state
		result_tokens = []
		new_tokens = tf.fill([batch_size, 1], self.start_token)
		done = tf.zeros([batch_size, 1], dtype=tf.bool) # stop flag for each elmt in batch

		# decoding
		for _ in range(max_length):
			dec_logit, dec_state = self.decoder(new_tokens, state=dec_state)
			new_tokens = self.sample_token(dec_logit, temperature)

			# Once a sequence is done it only produces 0-padding.
			done = done | (new_tokens == self.end_token)
			new_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), new_tokens)

			# Collect the generated tokens
			result_tokens.append(new_tokens)
			if tf.executing_eagerly() and tf.reduce_all(done):
				break

		# Convert the list of generates token ids to a list of strings.
		result_tokens = tf.concat(result_tokens, axis=-1)
		result_text = self.tokens_to_text(result_tokens)

		return result_text


class TranslatorAttGRU(tf.Module):

	def __init__(self, max_vocab_enc, max_vocab_dec, vocabfile_enc, vocabfile_dec,
				language, embedding_dim, hidden_dim, results_folder):

		# (1) Load tokenizer (convert input sentence to tokens)
		self.input_text_processor = load_encoder_vectorizer(max_vocab_enc,vocabfile_enc)
		output_text_processor = load_decoder_vectorizer(max_vocab_dec,vocabfile_dec,language)

		# (2) Load de-tokenizer (convert predicted tokens to words)
		dec_vocabfile2 = pathlib.Path(vocabfile_dec)
		text = dec_vocabfile2.read_text(encoding='utf-8')
		vocab_dec = text.splitlines()
		self.output_detokenizer = StringLookup( vocabulary=vocab_dec, invert=True)

		# (3) Special tokens and prediction mask
		# (mask to prohibit '', [UNK] or [BOS] during prediction)
		output_tokenizer = StringLookup( vocabulary=vocab_dec )
		self.start_token = output_tokenizer('[BOS]')
		self.end_token = output_tokenizer('[EOS]')
		token_mask_ids = output_tokenizer(['', '[UNK]', '[BOS]']).numpy()
		token_mask = np.zeros([len(vocab_dec)], dtype=np.bool)
		token_mask[np.array(token_mask_ids)] = True
		self.token_mask = token_mask

		# (4) Load encoder and decoder, and load weights
		self.encoder = EncoderGRU(max_vocab_enc, embedding_dim, hidden_dim)
		self.decoder = DecoderAttGRU(max_vocab_dec, embedding_dim, hidden_dim)

		# build before load weights
		enc_text = Input(shape=(1,), dtype=tf.string, name='enc')
		x = self.input_text_processor(enc_text)
		out, s = self.encoder(x)
		dec_text = Input(shape=(1,), dtype=tf.string, name='dec')
		x = output_text_processor(dec_text)
		x, _, _  = self.decoder(x, out, state = s )

		self.import_parameters(results_folder)


	def import_parameters(self, folderpath):
		with open(folderpath+'/weights_encoder.pickle', 'rb') as handle:
			enc_weights = pickle.load(handle)
		with open(folderpath+'/weights_decoder.pickle', 'rb') as handle:
			dec_weights = pickle.load(handle)
		self.encoder.set_weights(enc_weights)
		self.decoder.set_weights(dec_weights)

	def tokens_to_text(self, tokens):
		words = self.output_detokenizer(tokens)
		sentence = tf.strings.reduce_join(words, axis=1, separator=' ')
		sentence = tf.strings.strip(sentence)
		return sentence

	def sample_token(self, logits, temperature):

		# masking: where mask=True, take logit = -np.inf
		token_mask = self.token_mask[tf.newaxis, tf.newaxis, :] # extend array dimensions
		logits = tf.where(self.token_mask, -np.inf, logits)

		if temperature == 0.0: # greedy sampling
			new_tokens = tf.argmax(logits, axis=-1)
		else: # beam sampling
			logits = tf.squeeze(logits, axis=1)
			new_tokens = tf.random.categorical(logits/temperature, num_samples=1)

		return new_tokens

	def translate(self, input_text, max_length=50, temperature=1.0):

		batch_size = tf.shape(input_text)[0]
		
		# encoder
		input_tokens = self.input_text_processor(input_text)
		enc_output, enc_state = self.encoder(input_tokens)

		# decoding preparation
		dec_state = enc_state
		result_tokens = []
		new_tokens = tf.fill([batch_size, 1], self.start_token)
		done = tf.zeros([batch_size, 1], dtype=tf.bool) # stop flag for each elmt in batch

		# decoding
		for _ in range(max_length):
			dec_logit, dec_state, _ = self.decoder(new_tokens, enc_output, state=dec_state)
			new_tokens = self.sample_token(dec_logit, temperature)

			# Once a sequence is done it only produces 0-padding.
			done = done | (new_tokens == self.end_token)
			new_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), new_tokens)

			# Collect the generated tokens
			result_tokens.append(new_tokens)
			if tf.executing_eagerly() and tf.reduce_all(done):
				break

		# Convert the list of generates token ids to a list of strings.
		result_tokens = tf.concat(result_tokens, axis=-1)
		result_text = self.tokens_to_text(result_tokens)

		return result_text



# ----------------------------------------------------------------------