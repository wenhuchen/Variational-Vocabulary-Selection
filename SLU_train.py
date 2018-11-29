import os
import argparse
import logging
import sys
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
import math
from SLU_utils import createVocabulary, loadVocabulary, computeF1Score, DataProcessor

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

#Model and Vocab
parser.add_argument("--dataset", type=str, default=None, help="""Type 'atis' or 'snips' to use dataset provided by us or enter what ever you named your own dataset.
                Note, if you don't want to use this part, enter --dataset=''. It can not be None""")
parser.add_argument("--model_path", type=str, default='./SLU-model', help="Path to save model.")
parser.add_argument("--vocab_path", type=str, default='./SLU-vocab', help="Path to vocabulary files.")

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

createVocabulary(os.path.join(full_train_path, arg.input_file), os.path.join(arg.vocab_path, 'in_vocab'))
createVocabulary(os.path.join(full_train_path, arg.slot_file), os.path.join(arg.vocab_path, 'slot_vocab'))
createVocabulary(os.path.join(full_train_path, arg.intent_file), os.path.join(arg.vocab_path, 'intent_vocab'))

in_vocab = loadVocabulary(os.path.join(arg.vocab_path, 'in_vocab'))
slot_vocab = loadVocabulary(os.path.join(arg.vocab_path, 'slot_vocab'))
intent_vocab = loadVocabulary(os.path.join(arg.vocab_path, 'intent_vocab'))

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

class VarDropoutEmbedding(object):
    def __init__(self, input_size, layer_size, batch_size, name="embedding", is_training=True):
        self.name = name
        self.input_size = input_size
        self.layer_size = layer_size
        self.batch_size = batch_size
        self.is_training = is_training
        self.threshold = 3.0
        self.logdropout_init = tf.random_uniform_initializer(-1, 3)        
        self.embedding_mean = tf.get_variable(name, [self.input_size, self.layer_size])
        self.embedding_logdropout_ratio = tf.get_variable(name + "_ratio", [self.input_size, 1], initializer=self.logdropout_init)
        self.eps = tf.random_normal([self.batch_size, 1, self.layer_size], 0.0, 1.0)
        self.mask = tf.cast(tf.less(self.embedding_logdropout_ratio, self.threshold), tf.float32)
        self.sparsity = tf.nn.zero_fraction(self.mask)
      
    def __call__(self, input_data):
        if self.is_training:
            output_mean = tf.nn.embedding_lookup(self.embedding_mean, input_data)
            output_logdropout = tf.nn.embedding_lookup(self.embedding_logdropout_ratio, input_data)
            output_std = tf.exp(0.5 * output_logdropout) * output_mean
            output = output_mean + output_std * self.eps
        else:
            output = tf.nn.embedding_lookup(self.mask * self.embedding_mean, input_data)
        return output
    
    def zeroed_embedding(self):
        return self.embedding_mean * self.mask
    
    def regularizer(self):
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        log_alpha = self.embedding_logdropout_ratio
        KLD = -tf.reduce_sum(k1 * tf.sigmoid(k2 + k3 * log_alpha) - 0.5 * tf.nn.softplus(-log_alpha) - k1)
        return KLD    

def createModel(input_data, input_size, sequence_length, slot_size, intent_size, layer_size = 128, isTraining = True, batch_size=None):
    cell_fw = tf.contrib.rnn.BasicLSTMCell(layer_size)
    cell_bw = tf.contrib.rnn.BasicLSTMCell(layer_size)

    if isTraining == True:
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=0.5,
                                             output_keep_prob=0.5)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=0.5,
                                             output_keep_prob=0.5)

    #embedding = tf.get_variable('embedding', [input_size, layer_size])
    #inputs = tf.nn.embedding_lookup(embedding, input_data)
    embedding = VarDropoutEmbedding(input_size, layer_size, batch_size, is_training=isTraining)
    inputs = embedding(input_data)
    
    state_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=sequence_length, dtype=tf.float32)
    
    final_state = tf.concat([final_state[0][0], final_state[0][1], final_state[1][0], final_state[1][1]], 1)
    state_outputs = tf.concat([state_outputs[0], state_outputs[1]], 2)
    state_shape = state_outputs.get_shape()

    with tf.variable_scope('attention'):
        intent_input = final_state
        with tf.variable_scope('intent_attn'):
            attn_size = state_shape[2].value
            hidden = tf.expand_dims(state_outputs, 2)
            k = tf.get_variable("AttnW", [1, 1, attn_size, attn_size])
            hidden_features = tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
            v = tf.get_variable("AttnV", [attn_size])

            y = core_rnn_cell._linear(intent_input, attn_size, True)
            y = tf.reshape(y, [-1, 1, 1, attn_size])
            s = tf.reduce_sum(v*tf.tanh(hidden_features + y), [2,3])
            a = tf.nn.softmax(s)
            a = tf.expand_dims(a, -1)
            a = tf.expand_dims(a, -1)
            d = tf.reduce_sum(a * hidden, [1, 2])

            if add_final_state_to_intent == True:
                intent_output = tf.concat([d, intent_input], 1)
            else:
                intent_output = d

    with tf.variable_scope('intent_proj'):
        intent = core_rnn_cell._linear(intent_output, intent_size, True)

    outputs = [None, intent]
    return outputs, embedding

# Create Training Model
input_data = tf.placeholder(tf.int32, [None, None], name='inputs')
sequence_length = tf.placeholder(tf.int32, [None], name="sequence_length")
global_step = tf.Variable(0, trainable=False, name='global_step')
intent = tf.placeholder(tf.int32, [None], name='intent')

with tf.variable_scope('model'):
    training_outputs, embedding = createModel(input_data, len(in_vocab['vocab']), sequence_length, 
                                              len(slot_vocab['vocab']), len(intent_vocab['vocab']), 
                                              layer_size=arg.layer_size, batch_size=arg.batch_size)
    
def row_grouplasso(emebdding):
    col_sum = tf.reduce_sum(tf.square(embedding), axis=1)
    col_sum_sqrt = tf.sqrt(col_sum + 1e-8)
    return tf.reduce_sum(col_sum_sqrt)

intent_output = training_outputs[1]
with tf.variable_scope('intent_loss'):
    crossent =tf.nn.sparse_softmax_cross_entropy_with_logits(labels=intent, logits=intent_output)
    intent_loss = tf.reduce_sum(crossent) / tf.cast(arg.batch_size, tf.float32)

weight_decay = tf.placeholder(tf.float32, shape=(), name="weight_decay")

""" Lasso
reg_loss = weight_decay * row_grouplasso(embedding)
#row_sum = tf.reduce_sum(tf.abs(embedding), axis=1)
where_cond = tf.less(tf.abs(embedding), 0.0001)
zeroed_embedding = tf.assign(embedding, tf.where(where_cond,
                                          tf.zeros(tf.shape(embedding)),
                                          embedding))

sparsity = tf.nn.zero_fraction(tf.reduce_sum(zeroed_embedding, axis=1))
"""
reg_loss = weight_decay * embedding.regularizer()
sparsity = embedding.sparsity

params = tf.trainable_variables()
opt = tf.train.AdamOptimizer()

intent_params = []
slot_params = []
for p in params:
    if not 'slot_' in p.name:
        intent_params.append(p)
    if 'slot_' in p.name or 'bidirectional_rnn' in p.name or 'embedding' in p.name:
        slot_params.append(p)

gradients_intent = tf.gradients(intent_loss + reg_loss, intent_params)
clipped_gradients_intent, norm_intent = tf.clip_by_global_norm(gradients_intent, 5.0)
gradient_norm_intent = norm_intent
update_intent = opt.apply_gradients(zip(clipped_gradients_intent, intent_params), global_step=global_step)

training_outputs = [global_step, intent_loss, update_intent, gradient_norm_intent, reg_loss, sparsity]
inputs = [input_data, sequence_length, intent]

# Create Inference Model
with tf.variable_scope('model', reuse=True):
    inference_outputs, _ = createModel(input_data, len(in_vocab['vocab']), sequence_length, 
                                       len(slot_vocab['vocab']), len(intent_vocab['vocab']), 
                                       layer_size=arg.layer_size, isTraining=False, batch_size=arg.batch_size)

inference_intent_output = tf.nn.softmax(inference_outputs[1], name='intent_output')

inference_outputs = [inference_intent_output]
inference_inputs = [input_data, sequence_length]

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

saver = tf.train.Saver()

def valid(in_path, slot_path, intent_path):
    data_processor_valid = DataProcessor(in_path, slot_path, intent_path, in_vocab, slot_vocab, intent_vocab)

    pred_intents = []
    correct_intents = []
    slot_outputs = []
    correct_slots = []
    input_words = []

    #used to gate
    gate_seq = []
    while True:
        in_data, slot_data, slot_weight, length, intents, in_seq, slot_seq, intent_seq = data_processor_valid.get_batch(arg.batch_size)
        feed_dict = {input_data.name: in_data, sequence_length.name: length}
        ret = sess.run(inference_outputs, feed_dict)
        for i in ret[0]:
            pred_intents.append(np.argmax(i))
        for i in intents:
            correct_intents.append(i)

        if data_processor_valid.end == 1:
            break

    pred_intents = np.array(pred_intents)
    correct_intents = np.array(correct_intents)
    accuracy = (pred_intents==correct_intents)
    semantic_error = accuracy
    accuracy = accuracy.astype(float)
    accuracy = np.mean(accuracy)*100.0

    semantic_error = semantic_error.astype(float)
    semantic_error = np.mean(semantic_error)*100.0

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
        in_data, slot_data, slot_weight, length, intents,_,_,_ = data_processor.get_batch(arg.batch_size)
        cur_decay = min(math.pow(10, epochs // 3) * 0.00001, 0.01)
        feed_dict = {input_data.name: in_data, sequence_length.name: length, intent.name: intents, weight_decay: cur_decay}
        ret = sess.run(training_outputs, feed_dict)
        loss += np.mean(ret[1])
        cur_sparsity = ret[-1]

        line += arg.batch_size
        step = ret[0]
        num_loss += 1

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

            logging.info('Valid:')
            valid(os.path.join(full_valid_path, arg.input_file), os.path.join(full_valid_path, arg.slot_file), os.path.join(full_valid_path, arg.intent_file))

            logging.info('Test:')
            valid(os.path.join(full_test_path, arg.input_file), os.path.join(full_test_path, arg.slot_file), os.path.join(full_test_path, arg.intent_file))

            if epochs == arg.max_epochs:
                break
