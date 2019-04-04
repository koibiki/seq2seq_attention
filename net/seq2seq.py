import tensorflow as tf

from config import cfg
from net.base_net import BaseNet
from utils.seq_utils import preprocess_sentence, max_length, process_result


class Seq2Seq(BaseNet):

    def __init__(self, provider):
        # RNN Size
        rnn_size = 1024

        # Number of Layers
        num_layers = 1

        # Embedding Size
        encoding_embedding_size = 256
        decoding_embedding_size = 256

        self.provider = provider
        self.inp_lang = provider.inp_lang
        self.targ_lang = provider.targ_lang

        self.batch_size = provider.batch_size
        self.inputs = inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
        self.targets = targets = tf.placeholder(tf.int32, [None, None], name='targets')

        # 定义target序列最大长度（之后target_sequence_length和source_sequence_length会作为feed_dict的参数）
        self.target_sequence_length = target_sequence_length = tf.placeholder(tf.int32, (None,),
                                                                              name='target_sequence_length')
        max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
        self.source_sequence_length = source_sequence_length = tf.placeholder(tf.int32, (None,),
                                                                              name='source_sequence_length')

        training_decoder_output, predicting_decoder_output = self.seq2seq_model(self.batch_size,
                                                                                inputs,
                                                                                targets,
                                                                                target_sequence_length,
                                                                                max_target_sequence_length,
                                                                                source_sequence_length,
                                                                                self.inp_lang,
                                                                                self.targ_lang,
                                                                                encoding_embedding_size,
                                                                                decoding_embedding_size,
                                                                                rnn_size,
                                                                                num_layers)

        self.training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
        self.predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')

        self.masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32,
                                      name='masks')

        with tf.name_scope("optimization"):
            # Loss function
            self.cost = tf.contrib.seq2seq.sequence_loss(
                self.training_logits,
                targets,
                self.masks)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(cfg.LEARNING_RATE)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(self.cost)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            self.train_op = optimizer.apply_gradients(capped_gradients)

    def train(self):

        train_dataset = self.provider.train_dataset

        val_dataset = self.provider.val_dataset

        epochs = 50

        display_step = 100

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            train_iter = train_dataset.make_initializable_iterator()
            train_example = train_iter.get_next()

            val_iter = val_dataset.make_initializable_iterator()
            val_example = val_iter.get_next()

            sess.run(train_iter.initializer)
            sess.run(val_iter.initializer)

            saver = tf.train.Saver(max_to_keep=3)

            for epoch_i in range(epochs):

                for batch_i in range(self.provider.train_n_batch):
                    input_tensor, target_tensor, input_seq_len, target_seq_len = sess.run(train_example)

                    input_max_len = max(input_seq_len)
                    input_tensor = input_tensor[:, :input_max_len]

                    target_max_len = max(target_seq_len)
                    target_tensor = target_tensor[:, :target_max_len]

                    _, loss = sess.run(
                        [self.train_op, self.cost],
                        {self.inputs: input_tensor,
                         self.targets: target_tensor,
                         self.source_sequence_length: input_seq_len,
                         self.target_sequence_length: target_seq_len})

                    if batch_i % display_step == 0 or batch_i == self.provider.train_n_batch - 1:
                        print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f} '
                              .format(epoch_i,
                                      epochs,
                                      batch_i,
                                      self.provider.train_n_batch,
                                      loss))

                    saver.save(sess, "./checkpoint/net_{:d}.ckpt".format(epoch_i))
                    print('Model Saved epoch{:d}'.format(epoch_i))

    def predict(self, inputs):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(max_to_keep=3)

            model_file = tf.train.latest_checkpoint('./checkpoint')
            saver.restore(sess, model_file)

            inputs = [preprocess_sentence(in_seq) for in_seq in inputs]

            inputs = [inputs[0]]

            target_len = [6 for _ in range(len(inputs))]

            input_tensor = [[self.inp_lang.word2idx[s] for s in in_seq.split(' ')] for in_seq in inputs]

            input_len = [len(in_tensor) for in_tensor in input_tensor]

            max_len = max_length(input_tensor)

            input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,
                                                                         maxlen=max_len,
                                                                         padding='post',
                                                                         value=self.inp_lang.word2idx["<PAD>"])

            prediction = sess.run(self.predicting_logits, feed_dict={self.inputs: input_tensor,
                                                                     self.source_sequence_length: input_len,
                                                                     self.target_sequence_length: target_len})

            pred = [process_result(self.targ_lang, result) for result in prediction]

            print(pred)
