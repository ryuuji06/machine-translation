# clear existing user defined variables
for element in dir():
    if element[0:2] != "__":
        del globals()[element]

#import os
import pickle
import pathlib
from time import time

import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization, StringLookup

from seq2seq_models import TranslatorGRU, TranslatorAttGRU
from auxiliar_functions import load_translation_pairs
import matplotlib.pyplot as plt


# ==========================================================
#  WORD-LEVEL MACHINE TRANSLATION
#  PART III - PREDICTION
# ==========================================================
# exec(open('translation_predict.py').read())

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
#  S E T T I N G S
# ------------------------------------------------

# select model folder
#TEST = 'por_seq2seq_2' # file name
TEST = 'por_attention_2'
language = 'portuguese' # select language to translate
resultsfolder = 'results/'+TEST
vocabfolder = 'vocab/'+language



# ==================================================
print('\n(1) LOAD PARAMETERS, HISTORY AND MODEL')
# ==================================================

# Load parameters and history

with open(resultsfolder+'/params.pickle', 'rb') as handle:
	params = pickle.load(handle)
with open(resultsfolder+'/hist.pickle', 'rb') as handle:
	hist = pickle.load(handle)

MAX_VOCAB_ENG = params['MAX_VOCAB_ENG']
MAX_VOCAB_TRA = params['MAX_VOCAB_TRA']
EMBEDDING_DIM = params['EMBEDDING_DIM']
HIDDEN_DIM = params['HIDDEN_DIM']
BATCH_SIZE = params['BATCH_SIZE']

vocabfile1 = f'{vocabfolder}/eng{MAX_VOCAB_ENG}.txt'
vocabfile2 = f'{vocabfolder}/tra{MAX_VOCAB_TRA}.txt'

# Load prediction model (translator)

# translator = TranslatorGRU( MAX_VOCAB_ENG, MAX_VOCAB_TRA,
# 				vocabfile1, vocabfile2, language,
# 				EMBEDDING_DIM, HIDDEN_DIM, resultsfolder)
translator = TranslatorAttGRU( MAX_VOCAB_ENG, MAX_VOCAB_TRA,
				vocabfile1, vocabfile2, language,
				EMBEDDING_DIM, HIDDEN_DIM, resultsfolder)


# ==================================================
print('\n(2) TEST PREDICTION')
# ==================================================

# Random sample sentences
sentences = ["I'm coming home.",
			 "I don't know where I am.",
			 'I love you so much.',
			 'Did you say anything?',
			 'How dare you do that?',
			 'Has she received a recommendation letter from the professor?',
			 'I am not interested.',
			 'The sun did not rise today.',
			 'I hurt my left leg when I was younger.',
			 'This is none of your business.',
			 'Her business is not going well.',
			 'They have failed to overcome the economical crisis.']

# Predict
outputs = translator.translate(sentences)

for i in range(len(sentences)):
	print('Sentence ', i)
	print(' ', sentences[i])
	print(' ', outputs[i].numpy().decode())



# ==================================================
print('\n(3) PLOTS')
# ==================================================

plt.figure(1); plt.grid()
plt.plot(10*np.log10(hist[0]))
plt.plot(10*np.log10(hist[1]),'o')
plt.legend(['Training loss','Validation loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss (dB)')

plt.show()




# ----------------------------------------------------------------------