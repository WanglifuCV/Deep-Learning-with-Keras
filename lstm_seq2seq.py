# -*- coding:utf-8 -*-

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

    def __init__(self, data_dir, num_samples=None) -> None:
        data_loader = DataLoader(data_dir, num_samples=num_samples)
        self.encoder_input_data, self.decoder_input_data, self.decoder_target_data = data_loader.load_translation_data()

        self.encoder_inputs = keras.Input(shape=(None, data_loader.num_encoder_tokens))
        self.decoder_inputs = keras.Input(shape=(None, data_loader.num_decoder_tokens))
        self.num_encoder_tokens = data_loader.num_encoder_tokens
        self.num_decoder_tokens = data_loader.num_decoder_tokens
        self.model = self.build()


    def build(self):

        encoder_inputs = self.encoder_inputs
        encoder = keras.layers.LSTM(latent_dim, return_state=True)

        _, state_h, state_c = encoder(encoder_inputs)

        encoder_status = [state_h, state_c]

        decoder_inputs = self.decoder_inputs

        decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)

        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_status)

        decoder_dense = keras.layers.Dense(self.num_decoder_tokens, activation='softmax')

        decoder_outputs = decoder_dense(decoder_outputs)

        model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

        return model

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


class DataLoader(object):

    def __init__(self, data_dir, num_samples) -> None:
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.num_encoder_tokens = 0
        self.num_decoder_tokens = 0

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

        print('Number of samples: {}'.format(len(input_texts)))
        print('Number of unique input tokens: {}'.format(num_encoder_tokens))
        print('Number of unique output tokens: {}'.format(num_decoder_tokens))
        print('Max sequence length for inputs: {}'.format(max_encoder_seq_length))
        print('Max sequence length for outputs: {}'.format(max_decoder_seq_length))

        self.num_encoder_tokens = num_encoder_tokens
        self.num_decoder_tokens = num_decoder_tokens

        input_token_index = {char: i for i, char in enumerate(input_characters)}
        target_token_index = {char: i for i, char in enumerate(target_characters)}

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
    seq2seq_model = Seq2Seq(data_dir=data_path, num_samples=10000)
    seq2seq_model.model.summary()
    seq2seq_model.train(
        batch_size=64, 
        epochs=100
    )