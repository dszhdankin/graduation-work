import tensorflow as tf
import tensorflow_text as tf_text
import math
import time
import numpy as np


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


model = tf.keras.models.load_model(
    "saved_models/eng2sql_20", custom_objects={"standardize": standardize}
)
# model = tf.keras.models.load_model('saved_models/spa2eng_40')


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


# def translate_beam(spa_text, model, max_seq=100, beam_width=10):
#     spa_tokens = model.spa_text_processor([spa_text])  # Shape: (1, Ts)
#     spa_vectors = model.spa_embedding(spa_tokens, training=False)  # Shape: (1, Ts, embedding_dim)
#     spa_rnn_out, fhstate, fcstate, bhstate, bcstate = model.spa_rnn(spa_vectors,
#                                                                     training=False)  # Shape: (batch, rnn_output_dim)
#     spa_hstate = tf.concat([fhstate, bhstate], -1)
#     spa_cstate = tf.concat([fcstate, bcstate], -1)
#     state = [spa_hstate, spa_cstate]
#     print(spa_rnn_out.shape)
#
#     index_from_string = tf.keras.layers.StringLookup(
#         vocabulary=model.eng_text_processor.get_vocabulary(),
#         mask_token='')
#     trans = ['[START]']
#     vectors = []
#     output_sequences = [(['[START]'], [], state, 0.0)]
#     ended_sequences = []
#
#     for i in range(max_seq):
#         new_sequences = []
#         potential_sequences = []
#         for old_seq, old_vec, old_state, old_score in output_sequences:
#             token = index_from_string([[old_seq[i]]])
#             vector = model.eng_embedding(token, training=False)
#             old_vec.append(vector)
#             query = tf.concat(old_vec, axis=1)
#             context = model.attention(inputs=[query, spa_rnn_out], training=False)
#             trans_vector, hstate, cstate = model.eng_rnn(context[:, -1:, :], initial_state=old_state,
#                                                          training=False)
#             new_state = [hstate, cstate]
#             out = model.out(trans_vector)
#             out = tf.squeeze(out)
#
#             word_index = 0
#             for const_ts in out:
#                 t0 = time.process_time()
#                 cur_seq = old_seq + [model.eng_text_processor.get_vocabulary()[word_index]]
#                 cur_vec = old_vec
#                 print(time.process_time() - t0)
#                 t0 = time.process_time()
#                 cur_state = new_state
#                 cur_score = old_score + math.log(const_ts.numpy())
#                 tup = (cur_seq, cur_vec, cur_state, cur_score)
#                 word = model.eng_text_processor.get_vocabulary()[word_index]
#                 print(time.process_time() - t0)
#                 t0 = time.process_time()
#                 if word == '[END]':
#                     ended_sequences += [tup]
#                 else:
#                     potential_sequences += [tup]
#                 print(time.process_time() - t0)
#                 print(word_index)
#                 word_index += 1
#
#         new_sequences = sorted(potential_sequences, key=lambda val: val[3], reverse=True)
#         ended_sequences = sorted(ended_sequences, key=lambda val: val[3], reverse=True)
#         output_sequences = new_sequences[:beam_width]
#         ended_sequences = ended_sequences[:beam_width]
#
#         if len(output_sequences) == 0:
#             break
#
#     best_seq = None
#     if len(output_sequences) > 0:
#         best_seq = output_sequences[0]
#     if len(ended_sequences) > 0:
#         if best_seq is None:
#             best_seq = ended_sequences[0]
#         elif ended_sequences[0][3] > best_seq[3]:
#             best_seq = ended_sequences[0]
#
#     _, atts = model.attention(inputs=[best_seq[1], spa_rnn_out], return_attention_scores=True, training=False)
#     return ' '.join(best_seq[0][1:]), atts


print(translate('What is [table_10015132_16][Player]\' nationality', model)[0])
print(translate('What clu was in toronto [table_10015132_16][Years in Toronto]', model)[0])
print(translate('which club was in toronto [table_10015132_16][Years in Toronto]', model)[0])
print(translate('When the scoring rank was [table_10021158_3][Scoring rank], what was the best finish?',
                model)[0])
print(translate('who is the manufacturer for the model [table_10007452_3][model]?', model)[0])
print(translate('What is the current series where the [table_1000181_1][Notes]?', model)[0])
print(translate('Tell me what the notes are for [table_1000181_1][Current@slogan]', model)[0])

print('OK')