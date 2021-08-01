# Machine Translation with Bahdanau Attention Model

In this repository, I implement an automatic translation system from English to another language. It is currently trained to translate to portuguese, but it admits the translation to any language, provided available training data.

The **machine translation** task requires the computer to process an input sentence, in one language, and return as output the corresponding translation into some other language. For a long time there was much effort in solving this in a deterministic way, that is, by understanding the linguistics and grammatical rules of the languages and by formulating a representation for the meanings contained in a sentence. However, this require a large effort on formulating a complex language model.


Literature: how it was traditionally done? How is it done now?

RNN, sequence-to-sequence models, then sequence-to-sequence with attention, and more recently it
is performed with transformers

How target label is considered?


Dataset we use. Pre-trained embedding: GloVe 6B, that learns the vector representation of words unsupervisedly.

Describe more the model.

## About the model

## Sample result

<img src="https://github.com/ryuuji06/machine-translation/blob/main/images/ex_hist.png" width="400">


## References

[1] I. Sutskever, O. Vinyals and Q. Le. "Sequence to sequence learning with neural networks". In Advances in Neural Information Processing Systems (NIPS 2014).

[2] K. Cho, B. van Merrienboer, D. Bahdanau and Y. Bengio. "On the properties of neural machine translation: Encoder–Decoder approaches". In Eighth Workshop on Syntax, Semantics and Structure in Statistical Translation, 2014.

[3] D. Bahdanau, K. Cho and Y. Bengio. "Neural machine translation by jointly learning to align and translate". In International Conference on Learning Representations (ICLR)

[4] M. Luong, H. Pham, C.D. Manning. "Effective approaches to attention-based neural machine translation". 

[5] https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html

[x] Translation pairs dataset. http://www.manythings.org/anki/

[x] Pre-trained embeddings source. https://nlp.stanford.edu/projects/glove/

[] https://keras.io/examples/nlp/lstm_seq2seq/

[] https://www.tensorflow.org/text/tutorials/nmt_with_attention
