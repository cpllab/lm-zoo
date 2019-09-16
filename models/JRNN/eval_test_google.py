import os
import sys

import numpy as np
from six.moves import xrange
import tensorflow as tf

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
    sys.stderr.write('Recovering graph.\n')
    with tf.gfile.FastGFile(gd_file, 'r') as f:
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

    sys.stderr.write('Recovering checkpoint %s\n' % ckpt_file)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run('save/restore_all', {'save/Const:0': ckpt_file})
    sess.run(t['states_init'])

  return sess, t


def _EvalTestSents(input_file, vocab, output_file):
    targets = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
    weights = np.ones([BATCH_SIZE, NUM_TIMESTEPS], np.float32)

    # Load the model with the given pbtxt file and the checkpoint files
    sess, t = _LoadModel(FLAGS.pbtxt, FLAGS.ckpt)

    # Read intput file
    with open(input_file) as f:
        sents = f.readlines()

    result = []

    # print CSV header
    f = sys.stdout if output_file == "-" else open(output_file, "w")
    f.write("sentence_id\ttoken_id\ttoken\tsurprisal\n")

    for j in range(len(sents)):

        # Just so we know where things stand
        #if (j%10 == 0):
          #print(j/len(sents))

        inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
        char_ids_inputs = np.zeros( [BATCH_SIZE, NUM_TIMESTEPS, vocab.max_word_length], np.int32)

        sent = [vocab.word_to_id(w) for w in sents[j].split()]
        sent_char_ids = [vocab.word_to_char_ids(w) for w in sents[j].split()]

        samples = sent[:]
        char_ids_samples = sent_char_ids[:]

        # Total sentence surprisal
        total_surprisal = 0

        # First word in the sentence has a dummy surprisal of 0
        result.append(vocab.id_to_word(sent[0]) + "\t0.00")
        sess.run(t['states_init'])

        for n in range(len(sents[j].split(" "))-2):
            inputs[0, 0] = samples[0]
            char_ids_inputs[0, 0, :] = char_ids_samples[0]
            samples = samples[1:]
            char_ids_samples = char_ids_samples[1:]
            softmax = sess.run(t['softmax_out'],
                                 feed_dict={t['char_inputs_in']: char_ids_inputs,
                                            t['inputs_in']: inputs,
                                            t['targets_in']: targets,
                                            t['target_weights_in']: weights})

            surprisal = -1 * np.log2(softmax[0][sent[n+1]])
            total_surprisal += surprisal

            result.append("%i\t%i\t%s\t%f\n" % (j + 1, n + 1, vocab.id_to_word(sent[n+1]), surprisal))

    # Write result to output file
    if output_file != "-":
      f.close()

def main(unused_argv):
  vocab = data_utils.CharsVocabulary(FLAGS.vocab_file, MAX_WORD_LEN)
  _EvalTestSents(FLAGS.input_file, vocab, FLAGS.output_file)

if __name__ == '__main__':
  tf.app.run()
