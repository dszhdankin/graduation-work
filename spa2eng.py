import tensorflow as tf
import tensorflow_text as tf_text
import numpy as np
import math


def split(text):
    parts = tf.strings.split(text, sep='\t')
    return parts[0], parts[1]


dataset = tf.data.TextLineDataset(['datasets/spanish2english/spa-eng/spa.txt']).map(split)
eng_dataset = dataset.map(lambda eng, spa: eng)
spa_dataset = dataset.map(lambda eng, spa: spa)


def standardize(text):
    text = tf_text.normalize_utf8(text, normalization_form='NFKD')
    text = tf.strings.lower(text)

    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')

    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')

    text = tf.strings.strip(text)

    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text


eng_text_processor = tf.keras.layers.TextVectorization(standardize=standardize, max_tokens=5000)
spa_text_processor = tf.keras.layers.TextVectorization(standardize=standardize, max_tokens=5000)

eng_text_processor.adapt(eng_dataset.batch(128))
spa_text_processor.adapt(spa_dataset.batch(128))

print(eng_text_processor.get_vocabulary()[:40])
print(spa_text_processor.get_vocabulary()[:40])


class Spa2EngTranslator(tf.keras.Model):
    def __init__(self, eng_text_processor, spa_text_processor, unit=512):
        super().__init__()
        # Spanish
        self.spa_text_processor = spa_text_processor
        self.spa_vocabulary_size = len(spa_text_processor.get_vocabulary())
        self.spa_embedding = tf.keras.layers.Embedding(
            self.spa_vocabulary_size,
            output_dim=unit,
            mask_zero=True
        )
        self.spa_rnn = tf.keras.layers.Bidirectional(
            layer=tf.keras.layers.LSTM(int(unit / 2),
                                       return_sequences=True,
                                       return_state=True)
        )
        # Attention
        self.attention = tf.keras.layers.Attention()
        # English
        self.eng_text_processor = eng_text_processor
        self.eng_vocabulary_size = len(eng_text_processor.get_vocabulary())
        self.eng_embedding = tf.keras.layers.Embedding(
            self.eng_vocabulary_size,
            output_dim=unit,
            mask_zero=True
        )
        self.eng_rnn = tf.keras.layers.LSTM(unit, return_sequences=True, return_state=True)
        self.decoder_dropout = tf.keras.layers.Dropout(0.2)

        # Output
        self.out = tf.keras.layers.Dense(self.eng_vocabulary_size, activation='softmax')

    def call(self, eng_text, spa_text):
        # Shape: (batch, Ts)
        spa_tokens = self.spa_text_processor(spa_text)
        # Shape: (batch, Ts, embedding_dim)
        spa_vectors = self.spa_embedding(spa_tokens)
        # Shape: (batch, Ts, bi_rnn_output_dim), (batch, rnn_output_dim) ...
        spa_rnn_out, fhstate, fcstate, bhstate, bcstate = self.spa_rnn(spa_vectors)

        spa_hstate = tf.concat([fhstate, bhstate], -1)
        spa_cstate = tf.concat([fcstate, bcstate], -1)
        # print(fhstate.shape)
        # print(bhstate.shape)
        # print(fcstate.shape)
        # print(bcstate.shape)
        # print(spa_hstate.shape)
        # print(spa_cstate.shape)

        # Shape: (batch, Te-1)
        eng_tokens = self.eng_text_processor(eng_text)
        print(f'eng_tokens: {eng_tokens.shape}')
        # Shape: (batch, Te-1)
        expected = eng_tokens[:, 1:]
        print(f'expected: {expected.shape}')

        # Shape: (batch, Te-1)
        teacher_forcing = eng_tokens[:, :-1]
        print(f'teacher_forcing: {teacher_forcing.shape}')
        # Shape: (batch, Te-1, embedding_dim)
        eng_vectors = self.eng_embedding(teacher_forcing)
        print(f'eng_vectors: {eng_vectors.shape}')
        eng_in = self.attention(inputs=[eng_vectors, spa_rnn_out],
                                mask=[eng_vectors._keras_mask, spa_rnn_out._keras_mask])
        print(f'eng_in: {eng_in.shape}')

        # Shape: (batch, Te-1, rnn_output_dim)
        trans_vectors, _, _ = self.eng_rnn(eng_in, initial_state=[spa_hstate,
                                                                  spa_cstate])
        dropped_out = self.decoder_dropout(trans_vectors, training=True)

        # Shape: (batch, Te-1, eng_vocab_size)
        out = self.out(dropped_out)
        return out, expected, out._keras_mask


def train(epochs, model, batch=64, shuffle=1000):
    loss_fcn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE)
    opt = tf.keras.optimizers.Adam()
    losses = []
    ds = dataset.shuffle(shuffle).batch(batch).cache()
    for epoch in range(epochs):
        epoch_losses = []
        for eng_text, spa_text in ds:
            with tf.GradientTape() as tape:
                logits, expected, mask = model(eng_text, spa_text)
                loss = loss_fcn(expected, logits)
                loss = tf.ragged.boolean_mask(loss, mask)
                loss = tf.reduce_sum(loss) * (1. / batch)
                epoch_losses.append(loss.numpy())
                grads = tape.gradient(loss, model.trainable_weights)
                opt.apply_gradients(zip(grads, model.trainable_weights))
        losses.append(np.mean(epoch_losses))
        print('Trained epoch: {}; loss: {}'.format(epoch, losses[epoch]))
    #plt.plot(losses)
    #plt.xlabel('Epoch')
    #plt.ylabel('Losses')


def translate(spa_text, model, max_seq=100):
    spa_tokens = model.spa_text_processor([spa_text])  # Shape: (1, Ts)
    spa_vectors = model.spa_embedding(spa_tokens, training=False)  # Shape: (1, Ts, embedding_dim)
    spa_rnn_out, fhstate, fcstate, bhstate, bcstate = model.spa_rnn(spa_vectors,
                                                                    training=False)  # Shape: (batch, rnn_output_dim)
    spa_hstate = tf.concat([fhstate, bhstate], -1)
    spa_cstate = tf.concat([fcstate, bcstate], -1)
    state = [spa_hstate, spa_cstate]
    print(spa_rnn_out.shape)

    index_from_string = tf.keras.layers.StringLookup(
        vocabulary=model.eng_text_processor.get_vocabulary(),
        mask_token='')
    trans = ['[START]']
    vectors = []

    for i in range(max_seq):
        token = index_from_string([[trans[i]]])  # Shape: (1, 1)
        vector = model.eng_embedding(token, training=False)  # Shape: (1, 1, embedding_dim)
        vectors.append(vector)
        query = tf.concat(vectors, axis=1)
        context = model.attention(inputs=[query, spa_rnn_out], training=False)
        trans_vector, hstate, cstate = model.eng_rnn(context[:, -1:, :], initial_state=state,
                                                     training=False)  # Shape: (1, 1, rnn_output_dim), (1, rnn_output_dim), (1, rnn_output_dim)
        state = [hstate, cstate]
        out = model.out(trans_vector)  # Shape: (1, 1, eng_vocab_size)
        out = tf.squeeze(out)  # Shape: (eng_vocab_size,)
        word_index = tf.math.argmax(out)
        word = model.eng_text_processor.get_vocabulary()[word_index]
        trans.append(word)
        if word == '[END]':
            trans = trans[:-1]
            break
    _, atts = model.attention(inputs=[vectors, spa_rnn_out], return_attention_scores=True, training=False)
    return ' '.join(trans[1:]), atts


def translate_beam(spa_text, model, max_seq=100, beam_width=10):
    spa_tokens = model.spa_text_processor([spa_text])  # Shape: (1, Ts)
    spa_vectors = model.spa_embedding(spa_tokens, training=False)  # Shape: (1, Ts, embedding_dim)
    spa_rnn_out, fhstate, fcstate, bhstate, bcstate = model.spa_rnn(spa_vectors,
                                                                    training=False)  # Shape: (batch, rnn_output_dim)
    spa_hstate = tf.concat([fhstate, bhstate], -1)
    spa_cstate = tf.concat([fcstate, bcstate], -1)
    state = [spa_hstate, spa_cstate]
    print(spa_rnn_out.shape)

    index_from_string = tf.keras.layers.StringLookup(
        vocabulary=model.eng_text_processor.get_vocabulary(),
        mask_token='')
    trans = ['[START]']
    vectors = []
    output_sequences = [(['[START]'], [], state, 0.0)]
    ended_sequences = []

    for i in range(max_seq):
        new_sequences = []
        potential_sequences = []
        for old_seq, old_vec, old_state, old_score in output_sequences:
            token = index_from_string([[trans[i]]])
            vector = model.eng_embedding(token, training=False)
            old_vec.append(vector)
            query = tf.concat(old_vec, axis=1)
            context = model.attention(inputs=[query, spa_rnn_out], training=False)
            trans_vector, hstate, cstate = model.eng_rnn(context[:, -1:, :], initial_state=old_state,
                                                         training=False)
            new_state = [hstate, cstate]
            out = model.out(trans_vector)
            out = tf.squeeze(out)

            word_index = 0
            for const_ts in out:
                cur_seq = old_seq + [model.eng_text_processor.get_vocabulary()[word_index]]
                cur_vec = old_vec
                cur_state = new_state
                cur_score = old_score + math.log(const_ts.numpy())
                word = model.eng_text_processor.get_vocabulary()[word_index]
                if word == '[END]':
                    ended_sequences.append((cur_seq, cur_vec, cur_state, cur_score))
                else:
                    potential_sequences.append((cur_seq, cur_vec, cur_state, cur_score))
                word_index += 1

        new_sequences = sorted(potential_sequences, key=lambda val: val[3], reverse=True)
        ended_sequences = sorted(ended_sequences, key=lambda val: val[3], reverse=True)
        output_sequences = new_sequences[:beam_width]
        ended_sequences = ended_sequences[:beam_width]

        if len(output_sequences) == 0:
            break

    best_seq = None
    if len(output_sequences) > 0:
        best_seq = output_sequences[0]
    if len(ended_sequences) > 0:
        if best_seq is None:
            best_seq = ended_sequences[0]
        elif ended_sequences[0][3] > best_seq[3]:
            best_seq = ended_sequences[0]

    _, atts = model.attention(inputs=[best_seq[1], spa_rnn_out], return_attention_scores=True, training=False)
    return ' '.join(best_seq[0][1:]), atts

    # for i in range(max_seq):
    #     token = index_from_string([[trans[i]]])  # Shape: (1, 1)
    #     vector = model.eng_embedding(token, training=False)  # Shape: (1, 1, embedding_dim)
    #     vectors.append(vector)
    #     query = tf.concat(vectors, axis=1)
    #     context = model.attention(inputs=[query, spa_rnn_out], training=False)
    #     trans_vector, hstate, cstate = model.eng_rnn(context[:, -1:, :], initial_state=state,
    #                                                  training=False)  # Shape: (1, 1, rnn_output_dim), (1, rnn_output_dim), (1, rnn_output_dim)
    #     state = [hstate, cstate]
    #     out = model.out(trans_vector)  # Shape: (1, 1, eng_vocab_size)
    #     out = tf.squeeze(out)  # Shape: (eng_vocab_size,)
    #     word_index = tf.math.argmax(out)
    #     word = model.eng_text_processor.get_vocabulary()[word_index]
    #     trans.append(word)
    #     if word == '[END]':
    #         trans = trans[:-1]
    #         break
    # _, atts = model.attention(inputs=[vectors, spa_rnn_out], return_attention_scores=True, training=False)
    # return ' '.join(trans[1:]), atts


model = Spa2EngTranslator(unit=512, eng_text_processor=eng_text_processor, spa_text_processor=spa_text_processor)
train(40, model)
model.save('saved_models/spa2eng_dropout_40')

print(translate('Te amo', model)[0])
print(translate('Nunca te olvidaré', model)[0])
print(translate('Me siento terrible', model)[0])
print(translate('Hoy tengo la tarde libre, así que pienso ir al parque, sentarme bajo un árbol y leer un libro.',
                model)[0])
print(translate('Mi nueva casa está en una calle ancha que tiene muchos árboles.', model)[0])
print('OK')
