# Machine Translation with Bahdanau Attention Model

In this repository, I implement an automatic translation system from English to another language. It is currently trained to translate to Portuguese, but it admits the translation to any language, provided available training data.

The **machine translation** task requires the computer to process an input sentence, in one language, and return as output the corresponding translation into some other language. For a long time there was much effort in solving this in a deterministic way, that is, by understanding the linguistics and grammatical rules of the languages and by formulating a representation for the meanings contained in a sentence. However, this require a large effort on formulating a complex language model.

[1-3] started a new trend on solving this task with recurrent neural networks (RNN), approach that has become known as **neural machine translation**. Note that using a simple RNN to machine translation is not a trivial task, because a simple RNN outputs only a scalar or a sequence of same length as the input, which is not usually the case in text translation. In their works, they use networks composed of an encoder and a decoder. The encoder RNN processes the input sequence producing an intermediate fixed-length vector (known as context vector). The decoder RNN is initialized with the context vector, and outputs the translated sequence by word by word recursively, by feeding the output of each instant back to the decoder input.

This **encoder-decoder** model (also called sequence-to-sequence model) has problem with long sequences: as the only information the decoder has about the input sequence is its initialization, the decoder tends to drift away from the encoder information as it iterates the output. To remedy this problem, [4,5] proposed some modifications of the encoder-decoder, inserting an **attention mechanism**. Instead of outputing only a single vector, in their approach the encoder outputs an entire sequence (context sequence). Then, an attention mechanism (a kind of auxiliary fully connected network) weights the context sequence so that the decoder can focus on different parts of the encoder output as it produces recursively its output. These approaches attained better results in the machine translation task.

Posteriorly, [6] proposed the **transformer**, a network based solely on the attention mechanism that surprisingly dispenses any convolutional or recurrent structures. This architecture achieved the state-of-the-art performance on neural machine translation. In [7] you can find a more quick and informal yet very good explanation about the attention mechanisms.

## About this repository

In this repository, I implement an encoder-decoder RNN with the Bahdanau attention mechanism. The code implementations were based on [8,9]. After the words are tokenized, with vocabulary of 10,000 words (for English and for Portuguese), they are input to a 100-dim word embedding (in both encoder and decoder).
 - Encoder: embedding (10000, 100), GRU layer 512 units.
 - Decoder: 
   - processing target tokens: embedding (10000, 100), GRU layer 512 units;
   - processing encoder output and hidden GRU output: attention mechanism 512 units;
   - processing GRU output and attention output: dense layer 512 units, tanh activation, and a final dense layer (10000 units).

The word embedding of the encoder (for English) was pre-trained with GloVe 6B [10]. The dataset used for training was obtained from [11].


## Sample result

The figure below shows the process of the loss and the validation loss during training.

<img src="https://github.com/ryuuji06/machine-translation/blob/main/images/ex_hist.png" width="400">

In the following, I show some outputs of the network for some arbitrary input sentences. As prediction is done with beam search, the predicted sequence is not deterministic. Note that sentences from 1 to 6 have pretty much reliable translations. Sentences from 7 to 10, which are more uncommon and might be more distinct from the training examples, do not produced good translations, although the semantics of the original sentence is somewhat kept.

(1) `I'm coming home.` ==>
`estou chegando em casa .`
`estou vindo para casa .`

(2) `I don't know where I am.` ==>
`eu nao sei onde estou .`
`eu nao sei onde estou .`

(3) `I love you so much.` ==>
`amo muito voce .`
`amo bastante .`

(4) `Did you say anything?` ==>
`voce disse alguma coisa ?`
`voce disse algo ?`

(5) `How dare you do that?` ==>
`como voce ousa fazer isso ?`
`como ousa fazer isso ?`

(6) `I am not interested.` ==>
`eu nao estou interessado .`
`nao estou interessado .`

(7) `Has she received a recommendation letter from the professor?` ==>
`ela recebeu uma carta de asia do professor ?`
`ela recebeu uma carta de idade ?`

(8) `The sun did not rise today.` ==>
`o sol nao ficou de quinze .`
`o sol nao brilhava hoje .`

(9) `I hurt my left leg when I was younger.` ==>
`eu machuquei a perna do meu quando eu estava mais jovem .`
`eu entendi meu perna de vez em quando eu estava jovem .`

(10) `They have failed to overcome the economical crisis.` ==>
`eles nao tem estado de olho para a crise .`
`eles nao sao necessarios a media .`

  
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

[10] Pre-trained embeddings source. https://nlp.stanford.edu/projects/glove/

[11] Translation pairs dataset. http://www.manythings.org/anki/



