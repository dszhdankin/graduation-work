import tensorflow as tf
import tensorflow_text as tf_text
import numpy as np
import math

tf.config.run_functions_eagerly(True)


# ----------- Dataset extraction -----------
def split(text):
    parts = tf.strings.split(text, sep='\t')
    return parts[0], parts[1]


dataset = tf.data.TextLineDataset(['datasets/english2SQL-processed/eng2SQL_train.txt']).map(split)
eng_dataset = dataset.map(lambda eng, sql: eng)
sql_dataset = dataset.map(lambda eng, sql: sql)


def standardize(text):
    # Split accecented characters.
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    # Keep space, a to z, and select punctuation.
    # text = tf.strings.regex_replace(text, r'[^ 0-9a-z.?!,¿\]\[_\-]', '')
    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')

    # Strip whitespace.
    text = tf.strings.strip(text)

    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text


sql_text_processor = tf.keras.layers.TextVectorization(standardize=standardize, max_tokens=40000)
eng_text_processor = tf.keras.layers.TextVectorization(standardize=standardize, max_tokens=40000)

sql_text_processor.adapt(sql_dataset.batch(128))
eng_text_processor.adapt(eng_dataset.batch(128))
# ----------- Dataset extraction -----------


# ----------- Fixed embeddings extraction -----------
eng_vocabulary = eng_text_processor.get_vocabulary()
eng_vocab_size = len(eng_vocabulary)
embeddings_index = {}
with open('embeddings/glove.6B.100d.txt') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

    fixed_embedding_matrix = np.zeros((eng_vocab_size, 100))
    for i, word in enumerate(eng_vocabulary):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            fixed_embedding_matrix[i] = embedding_vector

    fixed_embedding = tf.keras.layers.Embedding(
        eng_vocab_size,
        100,
        embeddings_initializer=tf.keras.initializers.Constant(fixed_embedding_matrix),
        trainable=False,
        mask_zero=True)
# ----------- Fixed embeddings extraction -----------


# ----------- Model definition -----------
class Eng2SqlTranslator(tf.keras.Model):
    def __init__(self, eng_text_processor, sql_text_processor, fixed_embedding, unit=512):
        # English
        super().__init__()
        self.eng_text_processor = eng_text_processor
        self.eng_vocab_size = len(eng_text_processor.get_vocabulary())
        self.eng_embedding = tf.keras.layers.Embedding(
            self.eng_vocab_size,
            output_dim=int(unit/2),
            mask_zero=True
        )
        self.fixed_embedding = fixed_embedding
        self.eng_rnn = tf.keras.layers.Bidirectional(
            layer=tf.keras.layers.LSTM(
                int(unit/2),
                return_sequences=True,
                return_state=True
            )
        )
        # Attention
        self.attention = tf.keras.layers.Attention()
        # SQL
        self.sql_text_processor = sql_text_processor
        self.sql_vocab_size = len(sql_text_processor.get_vocabulary())
        self.sql_embedding = tf.keras.layers.Embedding(
            self.sql_vocab_size,
            output_dim=unit,
            mask_zero=True
        )
        self.sql_rnn = tf.keras.layers.LSTM(
            unit,
            return_state=True,
            return_sequences=True
        )
        # Final probabilities
        self.out = tf.keras.layers.Dense(self.sql_vocab_size, activation='softmax')

    def call(self, eng_text, sql_text, training=True):
        eng_tokens = self.eng_text_processor(eng_text)  # Shape: (batch, Ts)
        eng_vectors = self.eng_embedding(eng_tokens, training=training)  # Shape: (batch, Ts, embedding_dim)
        eng_fixed_vectors = self.fixed_embedding(eng_tokens)  # Shape: (batch, Ts, 100)
        eng_full_vectors = tf.concat([eng_vectors, eng_fixed_vectors], -1)  # Shape: (batch, Ts, embedding_dim+100)
        eng_rnn_out, fhstate, fcstate, bhstate, bcstate = self.eng_rnn(eng_full_vectors,
                                                                     training=training)  # Shape: (batch, Ts, bi_rnn_output_dim), (batch, rnn_output_dim) ...
        hstate = tf.concat([fhstate, bhstate], -1)
        cstate = tf.concat([fcstate, bcstate], -1)

        sql_tokens = self.sql_text_processor(sql_text)  # Shape: (batch, Te)
        expected = sql_tokens[:, 1:]  # Shape: (batch, Te-1)

        teacher_forcing = sql_tokens[:, :-1]  # Shape: (batch, Te-1)
        sql_vectors = self.sql_embedding(teacher_forcing, training=training)  # Shape: (batch, Te-1, embedding_dim)
        sql_in = self.attention(inputs=[sql_vectors, eng_rnn_out],
                                mask=[sql_vectors._keras_mask, eng_rnn_out._keras_mask], training=training)

        trans_vectors, _, _ = self.sql_rnn(sql_in,
                                           initial_state=[hstate, cstate],
                                           training=training)  # Shape: (batch, Te-1, rnn_output_dim)
        out = self.out(trans_vectors, training=training)
        return out, expected, out._keras_mask
        pass
# ----------- Model definition -----------


# ----------- Model training -----------
def train(dataset, epochs, model, batch=64, shuffle=1000):
    loss_fcn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE)
    opt = tf.keras.optimizers.Adam()
    losses = []
    ds = dataset.shuffle(shuffle).batch(batch).cache()
    for epoch in range(epochs):
        epoch_losses = []
        for eng_text, sql_text in ds:
            with tf.GradientTape() as tape:
                logits, expected, mask = model(eng_text, sql_text)
                loss = loss_fcn(expected, logits)
                loss = tf.ragged.boolean_mask(loss, mask)
                loss = tf.reduce_sum(loss) * (1. / batch)
                epoch_losses.append(loss.numpy())
                grads = tape.gradient(loss, model.trainable_weights)
                opt.apply_gradients(zip(grads, model.trainable_weights))
        losses.append(np.mean(epoch_losses))
        print('Trained epoch: {}; loss: {}'.format(epoch, losses[epoch]))
    # plt.plot(losses)
    # plt.xlabel('Epoch')
    # plt.ylabel('Losses')
# ----------- Model training -----------


# ----------- Translation method -----------
def translate(eng_text, model, max_seq=100):
    eng_tokens = model.eng_text_processor([eng_text])  # Shape: (1, Ts)
    eng_vectors = model.eng_embedding(eng_tokens, training=False)  # Shape: (1, Ts, embedding_dim)
    eng_fixed_vectors = model.fixed_embedding(eng_tokens)
    eng_full_vectors = tf.concat([eng_vectors, eng_fixed_vectors], -1)
    eng_rnn_out, fhstate, fcstate, bhstate, bcstate = model.eng_rnn(eng_full_vectors,
                                                                    training=False)  # Shape: (batch, rnn_output_dim)
    eng_hstate = tf.concat([fhstate, bhstate], -1)
    eng_cstate = tf.concat([fcstate, bcstate], -1)
    state = [eng_hstate, eng_cstate]

    index_from_string = tf.keras.layers.StringLookup(
        vocabulary=model.eng_text_processor.get_vocabulary(),
        mask_token='')
    trans = ['[START]']
    vectors = []

    for i in range(max_seq):
        token = index_from_string([[trans[i]]])  # Shape: (1, 1)
        vector = model.sql_embedding(token, training=False)  # Shape: (1, 1, embedding_dim)
        vectors.append(vector)
        query = tf.concat(vectors, axis=1)
        context = model.attention(inputs=[query, eng_rnn_out], training=False)
        trans_vector, hstate, cstate = model.sql_rnn(context[:, -1:, :], initial_state=state,
                                                     training=False)  # Shape: (1, 1, rnn_output_dim), (1, rnn_output_dim), (1, rnn_output_dim)
        state = [hstate, cstate]
        out = model.out(trans_vector)  # Shape: (1, 1, eng_vocab_size)
        out = tf.squeeze(out)  # Shape: (eng_vocab_size,)
        word_index = tf.math.argmax(out)
        word = model.sql_text_processor.get_vocabulary()[word_index]
        trans.append(word)
        if word == '[END]':
            trans = trans[:-1]
            break
    _, atts = model.attention(inputs=[vectors, eng_rnn_out], return_attention_scores=True, training=False)
    return ' '.join(trans[1:]), atts
# ----------- Translation method -----------


model = Eng2SqlTranslator(unit=512, eng_text_processor=eng_text_processor,
                          sql_text_processor=sql_text_processor, fixed_embedding=fixed_embedding)
train(dataset=dataset, epochs=20, model=model)
model.save('saved_models/eng2sql_20')

print(translate('What is [table_10015132_16][Player]\' nationality', model)[0])
print(translate('What clu was in toronto [table_10015132_16][Years in Toronto]', model)[0])
print(translate('which club was in toronto [table_10015132_16][Years in Toronto]', model)[0])
print(translate('When the scoring rank was [table_10021158_3][Scoring rank], what was the best finish?',
                model)[0])
# table_10007452_3
print(translate('who is the manufacturer for the model [table_10007452_3][Model]?', model)[0])

print('OK')
