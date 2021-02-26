# -*- coding:utf-8 -*-

from os import name
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops.gen_batch_ops import batch
from tensorflow.python.ops.gen_math_ops import mod

batch_size = 64 # batch size of training
epochs = 100 # Number of epochs to train for
latent_dim = 256 # Latent dimensionality of encoding space
num_samples = 10000 # Number of samples to train on
data_path = '/home/wanglifu/data/datasets/machine-translation/fra.txt'

# 代码要做的事情：https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
# 可参考的代码：https://keras.io/examples/nlp/lstm_seq2seq/


class Seq2Seq(object):

    def __init__(self, data_dir, num_samples=None, is_training=True, pretrain_model=None) -> None:
        data_loader = DataLoader(data_dir, num_samples=num_samples)
        self.encoder_input_data, self.decoder_input_data, self.decoder_target_data = data_loader.load_translation_data()
        self.data_loader = data_loader
        if is_training:            
            self.num_decoder_tokens = data_loader.num_decoder_tokens
            self.encoder_inputs = keras.Input(shape=(None, data_loader.num_encoder_tokens))
            self.decoder_inputs = keras.Input(shape=(None, data_loader.num_decoder_tokens))
            self.num_encoder_tokens = data_loader.num_encoder_tokens
            self.model = self.build_train()
        else:
            if pretrain_model is not None:
                self.encoder_model, self.decoder_model = self.load_model(pretrain_model_dir=pretrain_model)
            else:
                raise ValueError('"pretrain_model_dir" should not be NONE!')

    def build_train(self):
        # input_1
        encoder_inputs = self.encoder_inputs
        encoder = keras.layers.LSTM(latent_dim, return_state=True)

        _, state_h, state_c = encoder(encoder_inputs)

        # 这是编码层的最终输出
        encoder_status = [state_h, state_c]

        decoder_inputs = self.decoder_inputs

        decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)

        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_status)

        decoder_dense = keras.layers.Dense(self.num_decoder_tokens, activation='softmax')

        decoder_outputs = decoder_dense(decoder_outputs)

        model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

        return model

    def build_inference(self):
        """
        尝试构造的推理模型
        最后很可能没有用
        """

        encoder_inputs = self.encoder_inputs

        encoder = keras.layers.LSTM(latent_dim, return_state=True)

        _, state_h_enc, state_c_enc = encoder(encoder_inputs)

        encoder_states = [state_h_enc, state_c_enc]

        encoder_model = keras.Model(encoder_inputs, encoder_states)

        decoder_inputs = self.decoder_inputs

        decoder_state_input_h = keras.Input(shape=(latent_dim,), name='input_3')
        decoder_state_input_c = keras.Input(shape=(latent_dim,), name='input_4')

        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_lstm = keras.layers.LSTM(latent_dim, return_sequence=True, return_state=True)

        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs
        )

        decoder_states = [state_h_dec, state_c_dec]

        decoder_dense = keras.layers.Dense(self.num_decoder_tokens, activation='softmax')

        decoder_outputs = decoder_dense(decoder_outputs)

        decoder_model = keras.Model(
            [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
        )
        return encoder_model, decoder_model

    def train(self, batch_size=32, epochs=20):
        self.model.compile(
            optimizer='rmsprop', 
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model.fit(
            [self.encoder_input_data, self.decoder_input_data], 
            self.decoder_target_data,
            batch_size,
            epochs,
            validation_split=0.2
        )
        self.model.save('s2s')

    def load_model(self, pretrain_model_dir):
        """
        直接读取模型
        """
        model = keras.models.load_model(pretrain_model_dir)

        encoder_inputs = model.input[0]  # input_1
        _, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
        encoder_states = [state_h_enc, state_c_enc]
        encoder_model = keras.Model(encoder_inputs, encoder_states)

        decoder_inputs = model.input[1]  # input_2
        decoder_state_input_h = keras.Input(shape=(latent_dim,), name="input_3")
        decoder_state_input_c = keras.Input(shape=(latent_dim,), name="input_4")
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = model.layers[3]
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs
        )
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = model.layers[4]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = keras.Model(
            [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
        )
        return encoder_model, decoder_model

    # def inference(self, )

    def translate_text(self, input_text_idx):

        input_text = self.encoder_input_data[input_text_idx: input_text_idx + 1]
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_text)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.data_loader.num_decoder_tokens))

        # 
        target_token_index = self.data_loader.target_token_index
        target_seq[0, 0, target_token_index['\t']] = 1.0

        reverse_input_char_index = dict((i, char) for char, i in self.data_loader.input_token_index.items())
        reverse_target_char_index = dict((i, char) for char, i in self.data_loader.target_token_index.items())

        max_decoder_seq_len = self.data_loader.max_decoder_seq_length

        stop_condition = False

        decoded_sentence = ''

        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            if sampled_char == '\n' or len(decoded_sentence) > self.data_loader.max_decoder_seq_length:
                stop_condition = True

            target_seq = np.zeros((1, 1, self.data_loader.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.0

            status_value = [h ,c]

        return decoded_sentence

    def translate_text2(self, input_seq):

        reverse_input_char_index = dict((i, char) for char, i in self.data_loader.input_token_index.items())
        reverse_target_char_index = dict((i, char) for char, i in self.data_loader.target_token_index.items())

        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.data_loader.num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.data_loader.target_token_index["\t"]] = 1.0

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if sampled_char == "\n" or len(decoded_sentence) > self.data_loader.max_decoder_seq_length:
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.data_loader.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.0

            # Update states
            states_value = [h, c]
        return decoded_sentence


class DataLoader(object):
    def __init__(self, data_dir, num_samples) -> None:
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.num_encoder_tokens = 0
        self.num_decoder_tokens = 0
        self.input_token_index = None
        self.target_token_index = None
        self.max_encoder_seq_length = None
        self.max_decoder_seq_length = None

    def load_translation_data(self):
        # 拆解
        input_texts = []
        target_texts = []

        input_characters = set()
        target_characters = set()

        with open(self.data_dir, 'r', encoding='utf-8') as file_reader:
            lines = file_reader.readlines()

        if self.num_samples is None:
            self.num_samples = len(lines) - 1

        for line in lines[: min(self.num_samples, len(lines) - 1)]:
            input_text, target_text, _ = line.split('\t')

            # Use "tab" as the "start sequence" character
            # "\n" as "end sequence" character
            target_text = '\t' + target_text + '\n'

            input_texts.append(input_text.replace('\u202f', ''))
            target_texts.append(target_text.replace('\u202f', ''))

            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)

            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)

        input_characters = sorted(list(input_characters))
        target_characters = sorted(list(target_characters))

        num_encoder_tokens = len(input_characters)
        num_decoder_tokens = len(target_characters)

        max_encoder_seq_length = max([len(txt) for txt in input_texts])
        max_decoder_seq_length = max([len(txt) for txt in target_texts])

        self.max_encoder_seq_length = max_encoder_seq_length
        self.max_decoder_seq_length = max_decoder_seq_length

        print('Number of samples: {}'.format(len(input_texts)))
        print('Number of unique input tokens: {}'.format(num_encoder_tokens))
        print('Number of unique output tokens: {}'.format(num_decoder_tokens))
        print('Max sequence length for inputs: {}'.format(max_encoder_seq_length))
        print('Max sequence length for outputs: {}'.format(max_decoder_seq_length))

        self.num_encoder_tokens = num_encoder_tokens
        self.num_decoder_tokens = num_decoder_tokens

        input_token_index = {char: i for i, char in enumerate(input_characters)}
        target_token_index = {char: i for i, char in enumerate(target_characters)}

        self.input_token_index = input_token_index
        self.target_token_index = target_token_index

        # len(input_texts) 和 len(target_texts) 数量相同
        encoder_input_data = np.zeros(
            (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype=np.float32
        )

        decoder_input_data = np.zeros(
            (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype=np.float32
        )

        decoder_target_data = np.zeros(
            (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype=np.float32
        )

        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):

            # step 1
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, input_token_index[char]] = 1.0

            # step 2
            encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.0

            # step 3
            for t, char in enumerate(target_text):
                # 
                decoder_input_data[i, t, target_token_index[char]] = 1.0
                if t > 0:
                    decoder_target_data[i, t - 1, target_token_index[char]] = 1.0

            decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.0
            decoder_target_data[i, t:, target_token_index[' ']] = 1.0

        return encoder_input_data, decoder_input_data, decoder_target_data


if __name__ == '__main__':
    # seq2seq_model = Seq2Seq(data_dir=data_path, num_samples=10000)
    # seq2seq_model.model.summary()
    # seq2seq_model.train(
    #     batch_size=64, 
    #     epochs=100
    # )
    data_loader = DataLoader(data_dir=data_path, num_samples=10000)
    encoder_input_data, _, _ = data_loader.load_translation_data()
    seq2seq_model = Seq2Seq(
        data_dir=data_path, 
        num_samples=10000, 
        is_training=False, 
        pretrain_model='/home/wanglifu/learning/Deep-Learning-with-Keras/s2s'
        )

    for i in range(10000):
        # print(seq2seq_model.translate_text(i))
        input_seq = encoder_input_data[i: i+1]
        print(seq2seq_model.translate_text2(input_seq=input_seq))
        