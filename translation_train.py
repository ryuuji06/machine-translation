# clear existing user defined variables
for element in dir():
    if element[0:2] != "__":
        del globals()[element]

import os
import pathlib
import pickle
import numpy as np
from time import time

from tensorflow.data import Dataset
from tensorflow.keras.optimizers import Adam

from auxiliar_functions import load_translation_pairs
from seq2seq_models import EncoderGRU, DecoderAttGRU, BahdanauAttention
from seq2seq_models import TrainTranslatorGRU, TrainTranslatorAttGRU, MaskedLoss, CustomCheckpoint
#import matplotlib.pyplot as plt


# ==========================================================
#  WORD-LEVEL MACHINE TRANSLATION
#  PART II - PREPROCESSING AND TRAINING
# ==========================================================
# exec(open('translation_train.py').read())

# PROBLEM FORMULATION
# - given a sentence in English, translate it to French/Spanish

# english-french, simple encoder-decoder (character-level translation)
# https://keras.io/examples/nlp/lstm_seq2seq/
# dataset (Standford Natural Language Inference (SNLI) corpus)
# curl -O http://www.manythings.org/anki/fra-eng.zip
# unzip fra-eng.zip

# english-spanish
# https://www.tensorflow.org/text/tutorials/nmt_with_attention
# https://github.com/tensorflow/nmt
# SPANISH DATASET
# http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip
# http://www.manythings.org/anki/

# SUMMARY OF THE ALGORITHM (recurrent sequence-to-sequence model)
# (1) provide input sequence (sentence in English) and target sequence (sentence in Frech)
# (2) tokenize sequences to sequences of integers
# (3) encoder: turns input sequence to 2 state vectors
# (4) decoder: trained to turn the target sequences into the same sequence 
# but offset by one steptime in the future (teacher forcing). Initial state: state vectors from encoder
# (5) prediction (implementation issues)
# https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html


# ------------------------------------------------
#  T R A I N I N G   S E T T I N G S
# ------------------------------------------------

# TEST = 'por_seq2seq_1' # file name
# attention = False
# description = 'Translation to portuguese. Seq2seq model without Additive attention.'
# EMBEDDING_DIM = 100
# HIDDEN_DIM = 256

# TEST = 'por_seq2seq_1a' # file name
# attention = False
# description = 'Translation to portuguese. Seq2seq model without Additive attention. Test faster learning rate.'
# EMBEDDING_DIM = 100
# HIDDEN_DIM = 256

# TEST = 'por_seq2seq_2' # file name
# attention = False
# description = 'Translation to portuguese. Seq2seq model without Additive attention.'
# EMBEDDING_DIM = 100
# HIDDEN_DIM = 512

# TEST = 'por_seq2seq_3' # file name
# attention = False
# description = 'Translation to portuguese. Seq2seq model without Additive attention.'
# EMBEDDING_DIM = 100
# HIDDEN_DIM = 1024

# TEST = 'por_seq2seq_4' # file name
# attention = False
# description = 'Translation to portuguese. Seq2seq model without Additive attention.'
# EMBEDDING_DIM = 200
# HIDDEN_DIM = 1024

# TEST = 'por_attention_1' # file name
# attention = True
# description = 'Translation to portuguese. Seq2seq model with Additive attention.'
# EMBEDDING_DIM = 100
# HIDDEN_DIM = 256

TEST = 'por_attention_2' # file name
attention = True
description = 'Translation to portuguese. Seq2seq model with Additive attention.'
EMBEDDING_DIM = 100
HIDDEN_DIM = 512

# TEST = 'quick' # file name
# attention = True
# description = 'Translation to portuguese. Seq2seq model with Additive attention.'
# EMBEDDING_DIM = 100
# HIDDEN_DIM = 128 # 512

# --------------------------------------------------------------

language = 'portuguese' # select language to translate

# dataset parameters
VALID_SPLIT = 0.2
BATCH_SIZE = 256
EPOCHS = 30

# maximum vocabulary size
# (here, this is required to read the vocabulary file)
MAX_VOCAB_ENG = 10000
MAX_VOCAB_TRA = 10000


# Path to dataset
datapath = 'C:/datasets/translation_data'

vocabfolder = f'vocab/{language}'
vocabfile1 = f'{vocabfolder}/eng{MAX_VOCAB_ENG}.txt'
vocabfile2 = f'{vocabfolder}/tra{MAX_VOCAB_TRA}.txt'
if not os.path.exists(vocabfile1):
	raise AssertionError('Vocabulary did not learn for the specific language / vocabulary size.')
if not os.path.exists(vocabfile2):
	raise AssertionError('Vocabulary did not learn for the specific language / vocabulary size.')

#folder for results
resultsfolder = f'results/{TEST}'
if os.path.exists(resultsfolder):
	raise AssertionError('Folder already exists. Be sure to use a proper name for the test.')
else:
   os.makedirs(resultsfolder)



# =====================================================
print('\n(1) LOAD DATA (TRANSLATION PAIRS)')
# =====================================================
t1 = time()

# load two lists of texts, of corresponding translations
eng, tra = load_translation_pairs(datapath, language)

NUM_SAMPLES = len(eng)
TRAIN_BATCHES = int( NUM_SAMPLES*(1-VALID_SPLIT)/BATCH_SIZE )
NUM_TRAIN = TRAIN_BATCHES * BATCH_SIZE
NUM_VALID = NUM_SAMPLES-NUM_TRAIN
print(f'Number of translation pairs: {NUM_SAMPLES}')
print(f'Number of training samples: {NUM_TRAIN}')
print(f'Number of validation samples: {NUM_VALID}')
#print(f'Translation sample:\n{eng[10000]}\n{tra[10000]}')

# Convert to tf.dataset object
# (within b-string, special characters are shown as their byte representation)
# (we can print b-string by .decode())
full_dataset = Dataset.from_tensor_slices((eng, tra)).shuffle(NUM_SAMPLES)
full_dataset = full_dataset.batch(BATCH_SIZE)

train_dataset = full_dataset.take(TRAIN_BATCHES)
valid_dataset = full_dataset.skip(TRAIN_BATCHES)
del full_dataset
t2 = time(); print('Handling dataset: %.3fs'%(t2-t1)); t1 = t2

# for example_input_batch, example_target_batch in train_dataset.take(1):
# 	for i in range(5):
# 		print(f'{example_input_batch[i].numpy().decode()}\t{example_target_batch[i].numpy().decode()}')

params = {'NUM_TRAIN': NUM_TRAIN,
		  'NUM_VALID': NUM_VALID,
		  'MAX_VOCAB_ENG': MAX_VOCAB_ENG,
		  'MAX_VOCAB_TRA': MAX_VOCAB_TRA,
		  'BATCH_SIZE': BATCH_SIZE,
		  'EMBEDDING_DIM': EMBEDDING_DIM,
		  'HIDDEN_DIM': HIDDEN_DIM
		  }


descrip = f"""
-----------------------------------------
Model Configuration
-----------------------------------------

{description}

Number of training samples: {NUM_TRAIN}
Number of validation samples: {NUM_VALID}

Vocabulary size (english)    : {MAX_VOCAB_ENG}
Vocabulary size (translation): {MAX_VOCAB_TRA}
Batch size: {BATCH_SIZE}

Embedding dimension: {EMBEDDING_DIM}
Number of units in each layer: {HIDDEN_DIM}

"""

# Save parameters and text description
with open(resultsfolder+'/params.pickle', 'wb') as handle:
		pickle.dump(params, handle)
with open(resultsfolder+'/description.txt', 'w') as f:
	f.write(descrip)


# ==========================================================
print('\n(2) BUILD MODEL AND TRAIN')
# ==========================================================
# still to implement: update metrics (accuracy) within fit()

# Load model
if attention:
	translator = TrainTranslatorAttGRU(
    	EMBEDDING_DIM, HIDDEN_DIM,
    	MAX_VOCAB_ENG, MAX_VOCAB_TRA, vocabfile1, vocabfile2, language)
else:
	translator = TrainTranslatorGRU(
	    EMBEDDING_DIM, HIDDEN_DIM,
	    MAX_VOCAB_ENG, MAX_VOCAB_TRA, vocabfile1, vocabfile2, language)

# Configure the loss and optimizer
translator.compile( optimizer=Adam(), loss=MaskedLoss() )
# Saves best only, overwriting the former ones; encoder and decoder separately
check = CustomCheckpoint(resultsfolder)

# Training
hist1 = translator.fit( train_dataset, validation_data=valid_dataset,
    epochs=EPOCHS, callbacks=[check] )

# Save history (pickle and txt)
hist = [hist1.history['loss'], hist1.history['val_loss']]
with open(resultsfolder+'/hist.pickle', 'wb') as handle:
		pickle.dump(hist, handle)
with open(resultsfolder+'/hist.txt', 'w') as f:
	f.write('  loss\tval_loss\n')
	for i in range(EPOCHS):
		f.write('%6.3f\t%6.3f\n'%(hist[0][i],hist[1][i]))



# ----------------------------------------------------------------------