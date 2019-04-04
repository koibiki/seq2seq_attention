import tensorflow as tf
from tensorflow.python.layers.core import Dense


class BaseNet:

    def encoder(self, input_data, rnn_size, num_layers,
                source_sequence_length, inp_lang,
                encoding_embedding_size):
        source_vocab_size = len(inp_lang.word2idx.items())
        encoder_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)

        # RNN cell
        def get_lstm_cell(rnn_size):
            lstm_cell = tf.contrib.rnn.GRUCell(rnn_size)
            return lstm_cell

        cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])

        encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input,
                                                          sequence_length=source_sequence_length, dtype=tf.float32)

        return encoder_output, encoder_state

    def decoder(self, batch_size, targ_lang, decoding_embedding_size, num_layers, rnn_size,
                target_sequence_length, max_target_sequence_length, encoder_output, encoder_state,
                decoder_input):
        # 1. Embedding
        target_vocab_size = len(targ_lang.word2idx.items())
        decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
        decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)

        # 2. 构造Decoder中的RNN单元
        def get_decoder_cell(rnn_size):
            decoder_cell = tf.contrib.rnn.GRUCell(rnn_size)
            return decoder_cell

        cell = tf.contrib.rnn.MultiRNNCell([get_decoder_cell(rnn_size) for _ in range(num_layers)])

        attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size, encoder_output, target_sequence_length,
                                                         normalize=False,
                                                         name='BahdanauAttention')

        dec_cell = tf.contrib.seq2seq.AttentionWrapper(cell=cell, attention_mechanism=attn_mech)

        # 3. Output全连接层
        output_layer = Dense(target_vocab_size,
                             kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        # 4. Training decoder
        with tf.variable_scope("decode"):
            # 得到help对象
            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                                sequence_length=target_sequence_length,
                                                                time_major=False)
            # 构造decoder
            training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                               training_helper,
                                                               dec_cell.zero_state(batch_size, dtype=tf.float32)
                                                               .clone(cell_state=encoder_state),
                                                               output_layer)

            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                              impute_finished=True,
                                                                              maximum_iterations=max_target_sequence_length)
        # 5. Predicting decoder
        # 与training共享参数
        with tf.variable_scope("decode", reuse=True):
            # 创建一个常量tensor并复制为batch_size的大小
            start_tokens = tf.tile(tf.constant([targ_lang.word2idx['<GO>']], dtype=tf.int32), [batch_size],
                                   name='start_tokens')
            predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings,
                                                                         start_tokens,
                                                                         targ_lang.word2idx['<EOS>'])
            predicting_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                                 predicting_helper,
                                                                 dec_cell.zero_state(batch_size, dtype=tf.float32)
                                                                 .clone(cell_state=encoder_state),
                                                                 output_layer)

            predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                                impute_finished=True,
                                                                                maximum_iterations=max_target_sequence_length)

        return training_decoder_output, predicting_decoder_output

    def process_decoder_input(self, data, targ_lang, batch_size):
        '''
        补充<GO>，并移除最后一个字符
        '''
        # cut掉最后一个字符
        ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([batch_size, 1], targ_lang.word2idx['<GO>']), ending], 1)

        return decoder_input

    def seq2seq_model(self, batch_size, input_data, targets, target_sequence_length,
                      max_target_sequence_length, source_sequence_length,
                      inp_lang, targ_lang,
                      encoding_embedding_size, decoding_embedding_size,
                      rnn_size, num_layers):
        # 获取encoder的状态输出
        encoder_output, encoder_state = self.encoder(input_data,
                                                     rnn_size,
                                                     num_layers,
                                                     source_sequence_length,
                                                     inp_lang,
                                                     encoding_embedding_size)

        # 预处理后的decoder输入
        decoder_input = self.process_decoder_input(targets, targ_lang, batch_size)

        # 将状态向量与输入传递给decoder
        training_decoder_output, predicting_decoder_output = self.decoder(batch_size,
                                                                          targ_lang,
                                                                          decoding_embedding_size,
                                                                          num_layers,
                                                                          rnn_size,
                                                                          target_sequence_length,
                                                                          max_target_sequence_length,
                                                                          encoder_output,
                                                                          encoder_state,
                                                                          decoder_input)

        return training_decoder_output, predicting_decoder_output
