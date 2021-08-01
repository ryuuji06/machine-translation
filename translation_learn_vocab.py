# clear existing user defined variables
for element in dir():
    if element[0:2] != "__":
        del globals()[element]

import os
import pathlib
import pickle
import numpy as np
from time import time

from tensorflow import constant
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from auxiliar_functions import load_translation_pairs
from auxiliar_functions import english_standardization, spanish_standardization, french_standardization


# ==========================================================
#  WORD-LEVEL MACHINE TRANSLATION
#  PART I - LEARN VOCABULARY
# ==========================================================
# exec(open('translation_learn_vocab.py').read())

# PROBLEM FORMULATION
# - given a sentence in English, translate it to French/Spanish/Portuguese

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

# ----------------------------------------------


# from a given list of sentences in english `sentences`,
# learn the vocabulary (in occurence frequency order) and write it in a .txt file
def learn_eng(sentences, vocab_len, vocabfolder):

	vocabfile = vocabfolder + f'/eng{vocab_len}.txt'
	if os.path.exists(vocabfile):
		print('English vocabulary already learned.')
	
	else:
		t1 = time()
		input_text_processor = TextVectorization(
		    standardize=english_standardization,
		    max_tokens=vocab_len)
		
		print('Learning english vocabulary...')
		input_text_processor.adapt(sentences)
		vocab_eng = input_text_processor.get_vocabulary()
		
		with open(vocabfile, "w") as f:
			for line in vocab_eng[1:]:
				f.write('\n')
				f.write(line)
		t2 = time(); print('  vocabulary learned: %.3fs'%(t2-t1))
		print('  sample vocabulary')
		print(vocab_eng[:20])

# from a given list of sentences in the translation language `sentences`,
# learn the vocabulary (in occurence frequency order) and write it in a .txt file
def learn_tra(sentences, vocab_len, vocabfolder, language):

	vocabfile = vocabfolder + f'/tra{vocab_len}.txt'
	if os.path.exists(vocabfile):
		print('Translation vocabulary already learned.')
	else:
		t1 = time()
		if language == 'portuguese' or 'spanish':
			output_text_processor = TextVectorization(
		    	standardize=spanish_standardization,
		    	max_tokens=vocab_len)
		elif language == 'french':
			output_text_processor = TextVectorization(
		    	standardize=french_standardization,
		    	max_tokens=vocab_len)
		print('Learning translation vocabulary...')
		output_text_processor.adapt(sentences)
		vocab_tra = output_text_processor.get_vocabulary()
		
		with open(vocabfile, "w") as f:
			for line in vocab_tra[1:]:
				f.write('\n')
				f.write(line)
		t2 = time(); print('  vocabulary learned: %.3fs'%(t2-t1))
		print('  sample vocabulary')
		print(vocab_tra[:20])


# -------------------------------------
#   S E T T I N G S
# -------------------------------------

# select language to translate
language = 'portuguese' # portuguese, spanish, french

# maximum vocabulary size
MAX_VOCAB_ENG = 10000
MAX_VOCAB_TRA = 10000

datapath = 'C:/datasets/translation_data'

# folder for vocabulary
vocabfolder = f'vocab/{language}'
if not os.path.exists(vocabfolder):
	os.makedirs(vocabfolder)


# ==================================================
print('\n(1) LOAD DATA AND LEARN VOCABULARY')
# ==================================================
t1 = time()

# load two lists of texts, of corresponding translations
eng, tra = load_translation_pairs(datapath, language)

# learn vocabulary
learn_eng(eng, MAX_VOCAB_ENG, vocabfolder)
learn_tra(tra, MAX_VOCAB_TRA, vocabfolder, language)


# ==================================================
print('\n(2) LOAD TEST')
# ==================================================

vocab_file1 = pathlib.Path(vocabfolder)/f'eng{MAX_VOCAB_ENG}.txt'
vocab_file2 = pathlib.Path(vocabfolder)/f'tra{MAX_VOCAB_TRA}.txt'

text = vocab_file1.read_text(encoding='utf-8')
lines = text.splitlines() # list of strings
input_text_processor = TextVectorization(
    standardize=english_standardization,
    max_tokens=MAX_VOCAB_ENG,
    vocabulary=lines)

text = vocab_file2.read_text(encoding='utf-8')
lines = text.splitlines() # list of strings
if language == 'portuguese' or 'spanish':
	output_text_processor = TextVectorization(
    	standardize=spanish_standardization,
    	max_tokens=MAX_VOCAB_TRA,
    	vocabulary=lines)
elif language == 'french':
	output_text_processor = TextVectorization(
    	standardize=french_standardization,
    	max_tokens=MAX_VOCAB_TRA,
    	vocabulary=lines)

vocab_eng = np.array(input_text_processor.get_vocabulary())
vocab_tra = np.array(output_text_processor.get_vocabulary())


# TEST TOKENIZE AND DE-TOKENIZE
#example_text = constant('Pois você não estava certo: havia posto três vezes!! Como não vira os cento-e-um-cravos?')
example_text = constant(tra[10000])
example_tokens = output_text_processor(example_text)
tokens = vocab_tra[example_tokens.numpy()]

print('Example text: ', example_text.numpy().decode())
print('Example standardize: ', spanish_standardization(example_text).numpy().decode())
print('Example tokens: ', example_tokens.numpy())
print('De-tokenized: ',' '.join(tokens))
print()


# ----------------------------------------------------------------------