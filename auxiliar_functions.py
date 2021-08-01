
import pathlib
from tensorflow import strings as tf_strings

# remove accents (split accent from underlying letters)
# (there must be an equivalent within tensorflow or re)
import tensorflow_text as tf_text

# unicode normalization may be also be done as (only string argument)
# import unicodedata
# x = unicodedata.normalize('NFKD',x)


# -------------------------------------------
#  FUNCTION FOR LOADING DATA
# -------------------------------------------

def load_translation_pairs(path, language):

	if language == 'french':
		datafilepath = pathlib.Path(path)/'fra.txt'
	elif language == 'spanish':
		datafilepath = pathlib.Path(path)/'spa.txt'
	elif language == 'portuguese':
		datafilepath = pathlib.Path(path)/'por.txt'
	elif language == 'japanese':
		datafilepath = pathlib.Path(path)/'jpn.txt'

	text = datafilepath.read_text(encoding='utf-8') # creates a single large string
	lines = text.splitlines() # list of strings
	pairs = [line.split('\t') for line in lines] # separate languages into two strings per row

	# separate english and translations lists
	eng = [pairs[i][0] for i in range(len(lines))]
	tra = [pairs[i][1] for i in range(len(lines))]

	return eng, tra


# ----------------------------------------------------
#  TEXT STANDARDIZATION FUNCTIONS
# ----------------------------------------------------

# functions receives and returns tensor object

def english_standardization(text):
	# lower case
	text = tf_strings.lower(text)
	# substitute some expressions
	text = tf_strings.regex_replace(text, "i'm",  "i am")
	text = tf_strings.regex_replace(text, "'s",  " 's")
	text = tf_strings.regex_replace(text, "'ll", " will")
	text = tf_strings.regex_replace(text, "'ve", " have")
	text = tf_strings.regex_replace(text, "'re", " are")
	text = tf_strings.regex_replace(text, "'d",  " would")
	text = tf_strings.regex_replace(text, "won't", "will not")
	text = tf_strings.regex_replace(text, "can't", "can not")
	text = tf_strings.regex_replace(text, "n't", " not")
	text = tf_strings.regex_replace(text, "n'",  "ng")
	text = tf_strings.regex_replace(text, "'bout", "about")
	text = tf_strings.regex_replace(text, "'til", "until")
	# some punctuations
	text = tf_strings.regex_replace(text, '-', ' ')
	text = tf_strings.regex_replace(text, '[;:]', ',')
	# Keep space, a to z, and select punctuation.
	text = tf_strings.regex_replace(text, "[^ a-z.?!,¿']", '')
	# Add spaces around punctuation.
	text = tf_strings.regex_replace(text, '[.?!,¿]', r' \0 ')
	# Strip whitespace.
	text = tf_strings.strip(text)

	#text = tf_strings.join(['[BOS]', text, '[EOS]'], separator=' ')
	return text

# can also be used for portuguese
def spanish_standardization(text):
	# Split accecented characters.
	text = tf_text.normalize_utf8(text, 'NFKD')
	text = tf_strings.lower(text)
	# some punctuations
	text = tf_strings.regex_replace(text, '-', ' ')
	text = tf_strings.regex_replace(text, '[;:]', ',')
	# Keep space, a to z, and select punctuation.
	text = tf_strings.regex_replace(text, '[^ a-z.?!,¿]', '')
	# Add spaces around punctuation.
	text = tf_strings.regex_replace(text, '[.?!,¿]', r' \0 ')
	# Strip whitespace.
	text = tf_strings.strip(text)

	text = tf_strings.join(['[BOS]', text, '[EOS]'], separator=' ')
	return text

def french_standardization(text):
	# Split accecented characters.
	text = tf_text.normalize_utf8(text, 'NFKD')
	text = tf_strings.lower(text)
	# some punctuations
	text = tf_strings.regex_replace(text, "'", "' ")
	text = tf_strings.regex_replace(text, '-', ' ')
	text = tf_strings.regex_replace(text, '[;:]', ',')
	# Keep space, a to z, and select punctuation.
	text = tf_strings.regex_replace(text, '[^ a-z.?!,¿]', '')
	# Add spaces around punctuation.
	text = tf_strings.regex_replace(text, '[.?!,¿]', r' \0 ')
	# Strip whitespace.
	text = tf_strings.strip(text)

	text = tf_strings.join(['[BOS]', text, '[EOS]'], separator=' ')
	return text


# ----------------------------------------------------------------------