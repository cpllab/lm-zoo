import os
import sys

# Disable Tensorflow warning/info logs.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Disable Tensorflow deprecation warnings
try:
  from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
  from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0

import h5py
import numpy as np
from six.moves import xrange
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from google.protobuf import text_format
import data_utils

FLAGS = tf.flags.FLAGS
# General flags.
tf.flags.DEFINE_string('pbtxt', '',
                       'GraphDef proto text file used to construct model '
                       'structure.')
tf.flags.DEFINE_string('ckpt', '',
                       'Checkpoint directory used to fill model values.')
tf.flags.DEFINE_string('vocab_file', '', 'Vocabulary file.')
tf.flags.DEFINE_string('output_file', '',
                       'File to dump results.')
tf.flags.DEFINE_string('input_file', '',
                        'file of sentences to be evaluated')
tf.flags.DEFINE_string("mode", '', "One 'of 'surprisal', 'predictions'")

# For saving demo resources, use batch size 1 and step 1.
BATCH_SIZE = 1
NUM_TIMESTEPS = 1
MAX_WORD_LEN = 50


def _LoadModel(gd_file, ckpt_file):
  """Load the model from GraphDef and Checkpoint.

  Args:
    gd_file: GraphDef proto text file.
    ckpt_file: TensorFlow Checkpoint file.

  Returns:
    TensorFlow session and tensors dict.
  """
  with tf.Graph().as_default():
    with tf.gfile.GFile(gd_file, 'r') as f:
      s = f.read()
      gd = tf.GraphDef()
      text_format.Merge(s, gd)

    tf.logging.info('Recovering Graph %s', gd_file)
    t = {}
    [t['states_init'], t['lstm/lstm_0/control_dependency'],
     t['lstm/lstm_1/control_dependency'], t['softmax_out'], t['class_ids_out'],
     t['class_weights_out'], t['log_perplexity_out'], t['inputs_in'],
     t['targets_in'], t['target_weights_in'], t['char_inputs_in'],
     t['all_embs'], t['softmax_weights'], t['global_step']
    ] = tf.import_graph_def(gd, {}, ['states_init',
                                     'lstm/lstm_0/control_dependency:0',
                                     'lstm/lstm_1/control_dependency:0',
                                     'softmax_out:0',
                                     'class_ids_out:0',
                                     'class_weights_out:0',
                                     'log_perplexity_out:0',
                                     'inputs_in:0',
                                     'targets_in:0',
                                     'target_weights_in:0',
                                     'char_inputs_in:0',
                                     'all_embs_out:0',
                                     'Reshape_3:0',
                                     'global_step:0'], name='')

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run('save/restore_all', {'save/Const:0': ckpt_file})
    sess.run(t['states_init'])

  return sess, t


def get_predictions(sentences, model, sess, vocab):
  """
  Args:
    sentences: List of pre-tokenized lists of tokens
    model:
    sess:
    vocab: CharsVocabulary instance

  Yields lists of numpy arrays, one per sentence.
  """
  inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
  char_ids_inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS, vocab.max_word_length], np.int32)

  # Dummy inputs needed for the graph to compute
  targets = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
  target_weights = np.ones([BATCH_SIZE, NUM_TIMESTEPS], np.float32)

  for i, sentence in enumerate(sentences):
    sess.run(model["states_init"])

    # Compute token- and character-level vocabulary ID sequences
    sentence_ids = [vocab.word_to_id(w) for w in sentence]
    sentence_char_ids = [vocab.word_to_char_ids(w) for w in sentence]

    prev_word_id, prev_word_char_ids = None, None
    sentence_predictions = []
    for j, (word, word_id, char_ids) in enumerate(zip(sentence, sentence_ids, sentence_char_ids)):
      if j == 0:
        sentence_predictions.append(None)
      else:
        inputs[0, 0] = prev_word_id
        char_ids_inputs[0, 0, :] = prev_word_char_ids

        softmax = sess.run(model["softmax_out"],
                           feed_dict={model["inputs_in"]: inputs,
                                      model["char_inputs_in"]: char_ids_inputs,
                                      model["targets_in"]: targets,
                                      model["target_weights_in"]: target_weights})[0]

        # TODO JRNN softmax distribution size is greater than the vocabulary.
        # Why is that .. ?
        # In any case, let's just truncate and renorm to the actual vocab
        softmax = softmax[:vocab.size]
        softmax /= softmax.sum()
        softmax = np.log(softmax)

        sentence_predictions.append(softmax)

      prev_word_id = word_id
      prev_word_char_ids = char_ids

    yield sentence_predictions


def get_surprisals(sentences, model, sess, vocab):
  predictions = get_predictions(sentences, model, sess, vocab)
  for i, (sentence, sentence_preds) in enumerate(zip(sentences, predictions)):
    sentence_surprisals = []
    for j, (word_j, preds_j) in enumerate(zip(sentence, sentence_preds)):
      if preds_j is None:
        word_surprisal = 0.
      else:
        word_surprisal = -(preds_j[vocab.word_to_id(word_j)] / np.log(2))

      sentence_surprisals.append((word_j, word_surprisal))

    yield sentence_surprisals


def main(unused_argv):
  vocab = data_utils.CharsVocabulary(FLAGS.vocab_file, MAX_WORD_LEN)
  sess, model = _LoadModel(FLAGS.pbtxt, FLAGS.ckpt)

  with open(FLAGS.input_file) as inf:
    sentences = [line.strip().split(" ") for line in inf]

  if FLAGS.mode == "surprisal":
    outf = sys.stdout if FLAGS.output_file == "-" else open(output_file, "w")
    # Print TSV header
    outf.write("sentence_id\ttoken_id\ttoken\tsurprisal\n")

    surprisals = get_surprisals(sentences, model, sess, vocab)
    for i, (sentence, sentence_surps) in enumerate(zip(sentences, surprisals)):
      for j, (word, word_surp) in enumerate(sentence_surps):
        outf.write("%i\t%i\t%s\t%f\n" % (i + 1, j + 1, word, word_surp))

    outf.close()
  elif FLAGS.mode == "predictions":
    outf = h5py.File(FLAGS.output_file, "w")

    predictions = get_predictions(sentences, model, sess, vocab)
    for i, (sentence, sentence_preds) in enumerate(zip(sentences, predictions)):
        token_ids = [vocab.word_to_id(word) for word in sentence]

        # Skip the first word, which has null predictions
        sentence_preds = sentence_preds[1:]
        first_word_pred = np.ones_like(sentence_preds[0])
        first_word_pred /= first_word_pred.sum()
        sentence_preds = np.vstack([first_word_pred] + sentence_preds)

        group = outf.create_group("/sentence/%i" % i)
        group.create_dataset("predictions", data=sentence_preds)
        group.create_dataset("tokens", data=token_ids)

    vocab_encoded = np.array(vocab._id_to_word)
    vocab_encoded = np.char.encode(vocab_encoded, "utf-8")
    outf.create_dataset("/vocabulary", data=vocab_encoded)

    outf.close()
  else:
    raise ValueError("Unknown --mode %s" % FLAGS.mode)


if __name__ == '__main__':
  tf.app.run()
