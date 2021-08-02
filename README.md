# Machine Translation with Bahdanau Attention Model

In this repository, I implement an automatic translation system from English to another language. It is currently trained to translate to portuguese, but it admits the translation to any language, provided available training data.

The **machine translation** task requires the computer to process an input sentence, in one language, and return as output the corresponding translation into some other language. For a long time there was much effort in solving this in a deterministic way, that is, by understanding the linguistics and grammatical rules of the languages and by formulating a representation for the meanings contained in a sentence. However, this require a large effort on formulating a complex language model.

[1-3] started a new trend on solving this task with recurrent neural networks (RNN), approach that has become known as **neural machine translation**. Note that using a simple RNN to machine translation is not a trivial task, because a simple RNN outputs only a scalar or a sequence of same length as the input, which is not usually the case in text translation. In their works, they use networks composed of an encoder and a decoder. The encoder RNN processes the input sequence producing an intermediate fixed-length vector (known as context vector). The decoder RNN is initialized with the context vector, and outputs the translated sequence by word by word recursively, by feeding the output of each instant back to the decoder input.

This encoder-decoder model (also called sequence-to-sequence model) has problem with long sequences: as the only information the decoder has about the input sequence is its initialization, the decoder tends to drift away from the encoder information as it iterates the output. To remedy this problem, [4,5] proposed some modifications of the encoder-decoder, inserting an attention mechanism. Instead of outputing only a single vector, in their approach the encoder outputs an entire sequence (context sequence). Then, an attention mechanism (a kind of auxiliary fully connected network) weights the context sequence so that the decoder can focus on different parts of the encoder output as it produces recursively its output. These approaches attained better results in the machine translation task.

Posterior

## About the model

 Pre-trained embedding: GloVe 6B, that learns the vector representation of words unsupervisedly.

Dataset we use.

Describe more the model.

## Sample result

<img src="https://github.com/ryuuji06/machine-translation/blob/main/images/ex_hist.png" width="400">


## References

[1] N. Kalchbrenner and P. Blunsom. (2013). "Recurrent continuous translation models". In Proceedings of the ACL Conference on Empirical Methods in Natural Language Processing (EMNLP), p. 1700–1709, 2013.

[2] I. Sutskever, O. Vinyals and Q. Le. "Sequence to sequence learning with neural networks". In Advances in Neural Information Processing Systems (NIPS 2014).

[3] K. Cho, B. van Merrienboer, D. Bahdanau and Y. Bengio. "On the properties of neural machine translation: Encoder–Decoder approaches". In Eighth Workshop on Syntax, Semantics and Structure in Statistical Translation, 2014.

[4] D. Bahdanau, K. Cho and Y. Bengio. "Neural machine translation by jointly learning to align and translate". In International Conference on Learning Representations (ICLR), 2015.

[5] M. Luong, H. Pham, C.D. Manning. "Effective approaches to attention-based neural machine translation". Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP), p. 1412–1421, 2015.

[6] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A.N. Gomez, L. Kaiser, I. Polosukhin. "Attention is all you need". Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS'17), p. 6000–6010, 2017.



[6] https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html

[x] Translation pairs dataset. http://www.manythings.org/anki/

[x] Pre-trained embeddings source. https://nlp.stanford.edu/projects/glove/

[] https://keras.io/examples/nlp/lstm_seq2seq/

[] https://www.tensorflow.org/text/tutorials/nmt_with_attention
