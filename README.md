# Machine Translation with Bahdanau Attention Model

In this repository, I implement an automatic translation system from English to another language. It is currently trained to translate to Portuguese, but it admits the translation to any language, provided available training data.

The **machine translation** task requires the computer to process an input sentence, in one language, and return as output the corresponding translation into some other language. For a long time there was much effort in solving this in a deterministic way, that is, by understanding the linguistics and grammatical rules of the languages and by formulating a representation for the meanings contained in a sentence. However, this require a large effort on formulating a complex language model.

[1-3] started a new trend on solving this task with recurrent neural networks (RNN), approach that has become known as **neural machine translation**. Note that using a simple RNN to machine translation is not a trivial task, because a simple RNN outputs only a scalar or a sequence of same length as the input, which is not usually the case in text translation. In their works, they use networks composed of an encoder and a decoder. The encoder RNN processes the input sequence producing an intermediate fixed-length vector (known as context vector). The decoder RNN is initialized with the context vector, and outputs the translated sequence by word by word recursively, by feeding the output of each instant back to the decoder input.

This **encoder-decoder** model (also called sequence-to-sequence model) has problem with long sequences: as the only information the decoder has about the input sequence is its initialization, the decoder tends to drift away from the encoder information as it iterates the output. To remedy this problem, [4,5] proposed some modifications of the encoder-decoder, inserting an **attention mechanism**. Instead of outputing only a single vector, in their approach the encoder outputs an entire sequence (context sequence). Then, an attention mechanism (a kind of auxiliary fully connected network) weights the context sequence so that the decoder can focus on different parts of the encoder output as it produces recursively its output. These approaches attained better results in the machine translation task.

Posteriorly, [6] proposed the **transformer**, a network based solely on the attention mechanism that surprisingly dispenses any convolutional or recurrent structures. This architecture achieved the state-of-the-art performance on neural machine translation. In [7] you can find a more quick and informal yet very good explanation about the attention mechanisms.

## About this repository

In this repository, I implement an encoder-decoder RNN with the Bahdanau attention mechanism. The code implementations were based on [8,9]. After the words are tokenized, with vocabulary of 10,000 words (for English and for Portuguese), the input sequence is input to a 100-dim word embedding.

Encoder: embedding, GRU layer 512 units.

Decoder:

 Pre-trained embedding: GloVe 6B, that learns the vector representation of words unsupervisedly.

Dataset we use.

## Sample result

<img src="https://github.com/ryuuji06/machine-translation/blob/main/images/ex_hist.png" width="400">

`I'm coming home.`
`estou chegando em casa .`
`estou vindo para casa .`

`I don't know where I am.`
`eu nao sei onde estou .`
`eu nao sei onde estou .`

`I love you so much.`
`amo muito voce .`
`amo bastante .`

`Did you say anything?`
`voce disse alguma coisa ?`
`voce disse algo ?`

`How dare you do that?`
`como voce ousa fazer isso ?`
`como ousa fazer isso ?`

`Has she received a recommendation letter from the professor?`
`ela recebeu uma carta de asia do professor ?`
`ela recebeu uma carta de idade ?`

`I am not interested.`
`eu nao estou interessado .`
`nao estou interessado .`

`The sun did not rise today.`
`o sol nao ficou de quinze .`
`o sol nao brilhava hoje .`

`I hurt my left leg when I was younger.`
`eu machuquei a perna do meu quando eu estava mais jovem .`
`eu entendi meu perna de vez em quando eu estava jovem .`

`This is none of your business.`
`isto nao e de suas unica .`
`isto nao e da sua conta .`


  They have failed to overcome the economical crisis.
  eles nao tem estado de olho para a crise .
  eles nao sao necessarios a media .

  
## References

[1] N. Kalchbrenner and P. Blunsom. (2013). "Recurrent continuous translation models". In Proceedings of the ACL Conference on Empirical Methods in Natural Language Processing (EMNLP), p. 1700–1709, 2013.

[2] I. Sutskever, O. Vinyals and Q. Le. "Sequence to sequence learning with neural networks". In Advances in Neural Information Processing Systems (NIPS 2014).

[3] K. Cho, B. van Merrienboer, D. Bahdanau and Y. Bengio. "On the properties of neural machine translation: Encoder–Decoder approaches". In Eighth Workshop on Syntax, Semantics and Structure in Statistical Translation, 2014.

[4] D. Bahdanau, K. Cho and Y. Bengio. "Neural machine translation by jointly learning to align and translate". In International Conference on Learning Representations (ICLR), 2015.

[5] M. Luong, H. Pham, C.D. Manning. "Effective approaches to attention-based neural machine translation". Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP), p. 1412–1421, 2015.

[6] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A.N. Gomez, L. Kaiser, I. Polosukhin. "Attention is all you need". Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS'17), p. 6000–6010, 2017.

[7] https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html

[8] https://keras.io/examples/nlp/lstm_seq2seq/

[9] https://www.tensorflow.org/text/tutorials/nmt_with_attention


[x] Translation pairs dataset. http://www.manythings.org/anki/

[x] Pre-trained embeddings source. https://nlp.stanford.edu/projects/glove/

