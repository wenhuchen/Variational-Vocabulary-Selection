import os
import argparse
import logging
import sys
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
import math
from data_utils import *
from SLU_utils import createVocabulary, loadVocabulary, computeF1Score, DataProcessor
from models import VarDropoutEmbedding, NLUModel

parser = argparse.ArgumentParser()

#Network
parser.add_argument("--test", action="store_true", default=False, help="Whether to restore.")
parser.add_argument("--num_units", type=int, default=64, help="Network size.", dest='layer_size')
parser.add_argument("--model_type", type=str, default='full', help="""full(default) | intent_only
                                                                    full: full attention model
                                                                    intent_only: intent attention model""")

#Training Environment
parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
parser.add_argument("--max_epochs", type=int, default=100, help="Max epochs to train.")
parser.add_argument("--no_early_stop", action='store_false',dest='early_stop', help="Disable early stop, which is based on sentence level accuracy.")
parser.add_argument("--cutoff", type=int, default=10000, help="The cut off frequency")
#Model and Vocab
parser.add_argument("--dataset", type=str, default=None, help="""Type 'atis' or 'snips' to use dataset provided by us or enter what ever you named your own dataset.
                Note, if you don't want to use this part, enter --dataset=''. It can not be None""")
parser.add_argument("--model_path", type=str, default='./SLU-model', help="Path to save model.")
parser.add_argument("--vocab_path", type=str, default='./SLU-vocab', help="Path to vocabulary files.")
parser.add_argument("--variational", default=False, action='store_true', help="Whether to use variational training")
#Data
parser.add_argument("--train_data_path", type=str, default='train', help="Path to training data files.")
parser.add_argument("--test_data_path", type=str, default='test', help="Path to testing data files.")
parser.add_argument("--valid_data_path", type=str, default='valid', help="Path to validation data files.")
parser.add_argument("--input_file", type=str, default='seq.in', help="Input file name.")
parser.add_argument("--slot_file", type=str, default='seq.out', help="Slot file name.")
parser.add_argument("--intent_file", type=str, default='label', help="Intent file name.")

arg=parser.parse_args()

#Print arguments
for k,v in sorted(vars(arg).items()):
    print(k,'=',v)
print()

if arg.model_type == 'full':
    add_final_state_to_intent = True
    remove_slot_attn = False
elif arg.model_type == 'intent_only':
    add_final_state_to_intent = True
    remove_slot_attn = True
else:
    print('unknown model type!')
    exit(1)

#full path to data will be: ./data + dataset + train/test/valid
if arg.dataset == None:
    print('name of dataset can not be None')
    exit(1)
elif arg.dataset == 'snips':
    print('use snips dataset')
elif arg.dataset == 'atis':
    print('use atis dataset')
else:
    print('use own dataset: ',arg.dataset)

full_train_path = os.path.join('./SLU-data',arg.dataset,arg.train_data_path)
full_test_path = os.path.join('./SLU-data',arg.dataset,arg.test_data_path)
full_valid_path = os.path.join('./SLU-data',arg.dataset,arg.valid_data_path)

in_vocab = build_SLU_word_dict(arg.dataset, cutoff=arg.cutoff)
slot_vocab = build_SLU_word_dict(arg.dataset, cutoff=None)
intent_vocab = build_SLU_word_dict(arg.dataset, cutoff=None)

import pdb
pdb.set_trace()
#createVocabulary(os.path.join(full_train_path, arg.input_file), os.path.join(arg.vocab_path, 'in_vocab'))
#createVocabulary(os.path.join(full_train_path, arg.slot_file), os.path.join(arg.vocab_path, 'slot_vocab'))
#createVocabulary(os.path.join(full_train_path, arg.intent_file), os.path.join(arg.vocab_path, 'intent_vocab'))
#in_vocab = loadVocabulary(os.path.join(arg.vocab_path, 'in_vocab'))
#slot_vocab = loadVocabulary(os.path.join(arg.vocab_path, 'slot_vocab'))
#intent_vocab = loadVocabulary(os.path.join(arg.vocab_path, 'intent_vocab'))

def restore_trainables(sess, path):
    if path:
        assert tf.gfile.Exists(path)
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            variables_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, ckpt.model_checkpoint_path)
            print('Pre-trained model restored from %s' % path)
        else:
            print('Restoring pre-trained model from %s failed!' % path)
            exit()

with tf.variable_scope('model'):
    model = NLUModel(len(in_vocab), len(intent_vocab), layer_size=arg.layer_size, batch_size=arg.batch_size. isTraining=True)
with tf.variable_scope('model', reuse=True):
    test_model = NLUModel(len(in_vocab), len(intent_vocab), layer_size=arg.layer_size, batch_size=arg.batch_size. isTraining=False)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

saver = tf.train.Saver()

def valid(in_path, slot_path, intent_path):
    data_processor_valid = DataProcessor(in_path, slot_path, intent_path, in_vocab, slot_vocab, intent_vocab)

    sum_accuracy, cnt = 0, 0
    while True:
        in_data, slot_data, slot_weight, length, intents, in_seq, slot_seq, intent_seq = data_processor_valid.get_batch(arg.batch_size)
        feed_dict = {test_model.x: in_data, test_model.sequence_length: length}
        ret = sess.run(model.prediction, feed_dict)
        sum_accuracy += accuracy
        cnt += 1

        if data_processor_valid.end == 1:
            break

    test_accuracy = sum_accuracy / cnt
    #logging.info('slot f1: ' + str(f1))
    logging.info('intent accuracy: ' + str(accuracy))
    #logging.info('semantic error(intent, slots are all correct): ' + str(semantic_error))
    data_processor_valid.close()

# Start Training
model_name = "{}/model{}.ckpt".format(arg.model_path, arg.dataset)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    logging.info('Training Start')
    
    if tf.train.checkpoint_exists(model_name):
        saver.restore(sess, model_name)
        logging.info('Restored on Test:')    
        valid(os.path.join(full_test_path, arg.input_file), os.path.join(full_test_path, arg.slot_file), os.path.join(full_test_path, arg.intent_file))
        
    epochs = 0
    loss = 0.0
    data_processor = None
    line = 0
    num_loss = 0
    step = 0
    no_improve = 0

    #variables to store highest values among epochs, only use 'valid_err' for now
    valid_slot = 0
    test_slot = 0
    valid_intent = 0
    test_intent = 0
    valid_err = 0
    test_err = 0

    while True:
        if data_processor == None:
            data_processor = DataProcessor(os.path.join(full_train_path, arg.input_file), os.path.join(full_train_path, arg.slot_file), os.path.join(full_train_path, arg.intent_file), in_vocab, slot_vocab, intent_vocab)
        in_data, slot_data, slot_weight, length, intents, _, _, _ = data_processor.get_batch(arg.batch_size)
        #cur_decay = min(math.pow(10, epochs // 3) * 0.00001, 0.01)
        cur_decay, learning_rate = get_decay_rate(epochs)
                train_feed_dict = {
                    model.x: x_batch,
                    model.y: y_batch,
                    model.weight_decay: cur_decay,
                    model.learning_rate: learning_rate,
                    model.threshold: 3.0,
                    model.l1_threshold: 1e-4
                }

        if arg.variational:
            _, step, loss, reg_loss, sparsity = sess.run([model.optimizer, model.global_step, model.cross_entropy, model.reg_loss, model.sparsity], feed_dict=train_feed_dict)
        elif arg.l1:
            _, step, loss, reg_loss, sparsity = sess.run([model.optimizer, model.global_step, model.cross_entropy, model.reg_loss, model.sparsity], feed_dict=train_feed_dict)
        else:
            _, step, loss, reg_loss, sparsity = sess.run([model.optimizer, model.global_step, model.cross_entropy, model.reg_loss, model.sparsity], feed_dict=train_feed_dict)

        if step % 100 == 0:
            print("epoch {0}: KL_decay {1}: step {2}: cross_entropy = {3}: reg_loss = {4}, sparsity = {5}: full_vocab = {6}: remaining_vocab: {7}".format(epochs, cur_decay, step, loss, reg_loss, sparsity, vocabulary_size, int((1 - sparsity) * vocabulary_size)))


        if data_processor.end == 1:
            line = 0
            data_processor.close()
            data_processor = None
            epochs += 1
            logging.info('Step: ' + str(step))
            logging.info('Epochs: ' + str(epochs))
            logging.info('Loss: ' + str(loss/num_loss))
            logging.info('Embedding Sparsity: ' + str(cur_sparsity))
            np.save("embedding-{}".format(arg.dataset), sess.run(embedding.zeroed_embedding()))
            
            num_loss = 0
            loss = 0.0

            #save_path = os.path.join(arg.model_path,'_step_' + str(step) + '_epochs_' + str(epochs) + '.ckpt')
            saver.save(sess, model_name)

            
            if epochs == arg.max_epochs:
                break
