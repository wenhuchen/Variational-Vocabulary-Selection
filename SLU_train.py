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
from models.models import NLUModel
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

parser = argparse.ArgumentParser()

parser.add_argument("--test", action="store_true", default=False, help="Whether to restore.")
parser.add_argument("--num_units", type=int, default=64, help="Network size.", dest='layer_size')
parser.add_argument("--model_type", type=str, default='full', help="""full(default) | intent_only
                                                                    full: full attention model
                                                                    intent_only: intent attention model""")
parser.add_argument("--id", type=str, default="baseline", help="The id of the model")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
parser.add_argument("--max_epochs", type=int, default=100, help="Max epochs to train.")
parser.add_argument("--cutoff", type=int, default=10000, help="The cut off frequency")
parser.add_argument("--dataset", type=str, default=None, help="""Type 'atis' or 'snips' to use dataset provided by us or enter what ever you named your own dataset. Note, if you don't want to use this part, enter --dataset=''. It can not be None""")
parser.add_argument("--model_path", type=str, default='./SLU-model', help="Path to save model.")
parser.add_argument("--vocab_path", type=str, default='./SLU-vocab', help="Path to vocabulary files.")
parser.add_argument("--variational", default=False, action='store_true', help="Whether to use variational training")
parser.add_argument("--l1", default=False, action="store_true")
parser.add_argument("--compress", default=False, action="store_true")
parser.add_argument("--bound", default=False, action="store_true")
parser.add_argument("--visualize", default=False, action="store_true")
parser.add_argument("--evaluate", default=False, action="store_true")
parser.add_argument("--tf_idf", default=False, action="store_true")
parser.add_argument("--threshold", default=3, type=float)
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
    FULL_VOCAB = 11000
elif arg.dataset == 'atis':
    print('use atis dataset')
    FULL_VOCAB = 724
else:
    print('use own dataset: ',arg.dataset)

full_train_path = os.path.join('./SLU-data',arg.dataset,arg.train_data_path)
full_test_path = os.path.join('./SLU-data',arg.dataset,arg.test_data_path)
full_valid_path = os.path.join('./SLU-data',arg.dataset,arg.valid_data_path)
test_in_path = os.path.join(full_test_path, arg.input_file)
test_slot_path = os.path.join(full_test_path, arg.slot_file)
test_intent_path = os.path.join(full_test_path, arg.intent_file)

in_vocab = build_SLU_word_dict(os.path.join(full_train_path, "seq.in"), '{}-data-{}'.format(arg.dataset, arg.cutoff), cutoff=arg.cutoff, stopword=True)
slot_vocab = build_SLU_word_dict(os.path.join(full_train_path, "seq.out"), '{}-slot'.format(arg.dataset), cutoff=None, stopword=True)
intent_vocab = build_SLU_word_dict(os.path.join(full_train_path, "label"), '{}-intent'.format(arg.dataset), cutoff=None, stopword=True)
vocabulary_size = len(in_vocab['vocab'])

with tf.variable_scope('model'):
    model = NLUModel(vocabulary_size, len(intent_vocab['vocab']), layer_size=arg.layer_size, batch_size=arg.batch_size, is_training=True, variational=arg.variational, l1=arg.l1, compress=arg.compress)
with tf.variable_scope('model', reuse=True):
    test_model = NLUModel(vocabulary_size, len(intent_vocab['vocab']), layer_size=arg.layer_size, batch_size=arg.batch_size, is_training=False, variational=arg.variational, l1=arg.l1, compress=arg.compress)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def valid(in_path, slot_path, intent_path):
    data_processor_valid = DataProcessor(in_path, slot_path, intent_path, in_vocab, slot_vocab, intent_vocab)
    sum_accuracy, cnt = 0, 0
    while True:
        in_data, slot_data, slot_weight, length, intents, in_seq, slot_seq, intent_seq = data_processor_valid.get_batch(arg.batch_size)
        train_feed_dict = {
            test_model.x: in_data,
            test_model.y: intents,
            test_model.sequence_length: length,
            test_model.threshold: 3.0,
            test_model.l1_threshold: 1e-4
        }
        _, accuracy = sess.run([test_model.predictions, test_model.accuracy], feed_dict=train_feed_dict) 
        sum_accuracy += accuracy
        cnt += 1
        if data_processor_valid.end == 1:
            break

    test_accuracy = sum_accuracy / cnt
    logging.info('intent accuracy: ' + str(test_accuracy))
    data_processor_valid.close()

# Start Training
if arg.variational:
	model_folder = "{}_models/{}_{}_variational".format(arg.dataset, arg.id, arg.layer_size)
elif arg.l1:
	model_folder = "{}_models/{}_{}_l1".format(arg.dataset, arg.id, arg.layer_size)
else:
	model_folder = "{}_models/{}_{}".format(arg.dataset, arg.id, arg.layer_size)
if not os.path.exists(model_folder):
	os.mkdir(model_folder)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    variables_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    saver = tf.train.Saver(variables_to_restore)
    logging.info('Training Start')
    
    model_name = os.path.join(model_folder, "model.ckpt")
    if tf.train.checkpoint_exists(model_name):
		saver.restore(sess, model_name)
		logging.info('Restored from previous model: {}'.format(model_name))
    
    if arg.visualize and arg.variational:
        ratios = sess.run(model.embedding.embedding_logdropout_ratio).squeeze()
        ranks = np.argsort(ratios)
        ws = [in_vocab['rev'][w] for w in ranks[:100]]
        with open('/tmp/words.txt', 'w') as fs:
            print >> fs, " ".join(ws)
        ws = [in_vocab['rev'][w] for w in range(100)]
        with open('/tmp/words_freq.txt', 'w') as fs:
            print >> fs, " ".join(ws)
        """
        ratios.sort()
        t = ratios[int(vocabulary_size * 0.01)]
        emb = sess.run(test_model.embedding_matrix, feed_dict={test_model.threshold: t})
        emb = (np.transpose(np.abs(emb))[:, :4000] > 0).astype('float32')
        plt.imshow(emb, cmap=cm.Greys)
        plt.savefig("images/{}_emb.eps".format(arg.dataset), format='eps', dpi=100)
        """
        sys.exit(1)

    if arg.evaluate:
        if arg.variational:
        #for t in np.linspace(-3, 3, 10):
            mask, emb, ratios = sess.run([model.mask, model.embedding.embedding_mean, model.embedding.embedding_logdropout_ratio], feed_dict={model.threshold: arg.threshold})
            #new_word_dict = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
            new_in_vocab = {"vocab": {"_PAD": 0, "_UNK": 1}, "rev": {}}
            emb_mean = [emb[0], emb[1]]
            emb_logdropout_ratio = [0., 0.]
            for i in range(vocabulary_size):
                if mask[i, 0] > 0:
                    new_in_vocab['vocab'][in_vocab['rev'][i]] = len(new_in_vocab)
                    emb_mean.append(emb[i])
                    emb_logdropout_ratio.append(ratios[i])
            new_vocabulary_size = len(emb_mean)
            emb_mean = np.vstack(emb_mean)
            emb_logdropout_ratio = np.vstack(emb_logdropout_ratio)
            data_processor_valid = DataProcessor(test_in_path, test_slot_path, test_intent_path, new_in_vocab, slot_vocab, intent_vocab)

            with tf.variable_scope("new_model", reuse=tf.AUTO_REUSE):
                new_test_model = NLUModel(new_vocabulary_size, len(intent_vocab['vocab']), layer_size=arg.layer_size, batch_size=arg.batch_size, is_training=False, variational=False, l1=arg.l1, compress=arg.compress)
            new_variables_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="new_model")
            init_ops = []
            for v in new_variables_to_restore:
                if v.op.name == "new_model/embedding":
                    init_ops.append(tf.assign(v, emb_mean))
                elif v.op.name ==  "new_model/embedding_ratio":
                    init_ops.append(tf.assign(v, emb_logdropout_ratio))
                else:
                    for v1 in variables_to_restore:
                        if v.op.name.split('/')[1:] == v1.op.name.split('/')[1:]:
                            init_ops.append(tf.assign(v, v1))

            for _ in init_ops:
                sess.run(_)

            sum_accuracy, cnt = 0, 0
            inf_time = 0
            while True:
                in_data, slot_data, slot_weight, length, intents, in_seq, slot_seq, intent_seq = data_processor_valid.get_batch(arg.batch_size)
                train_feed_dict = {
                    new_test_model.x: in_data,
                    new_test_model.y: intents,
                    new_test_model.sequence_length: length
                }
                start_time = time.time()            
                _, accuracy = sess.run([new_test_model.predictions, new_test_model.accuracy], feed_dict=train_feed_dict) 
                inf_time += time.time() - start_time
                sum_accuracy += accuracy
                cnt += 1
                if data_processor_valid.end == 1:
                    break

            test_accuracy = sum_accuracy / cnt

            whole_size = sum([tf.size(_) for _ in new_variables_to_restore])
            print("Accuracy = {} with vocabulary {} inference used {} for model size {}".format(test_accuracy, new_vocabulary_size, inf_time, sess.run(whole_size)))
        else:
            data_processor_valid = DataProcessor(test_in_path, test_slot_path, test_intent_path, in_vocab, slot_vocab, intent_vocab)
            while True:
                in_data, slot_data, slot_weight, length, intents, in_seq, slot_seq, intent_seq = data_processor_valid.get_batch(arg.batch_size)
                train_feed_dict = {
                    test_model.x: in_data,
                    test_model.y: intents,
                    test_model.sequence_length: length
                }
                start_time = time.time()            
                _, accuracy = sess.run([test_model.predictions, test_model.accuracy], feed_dict=train_feed_dict) 
                inf_time += time.time() - start_time
                sum_accuracy += accuracy
                cnt += 1
                if data_processor_valid.end == 1:
                    break

            test_accuracy = sum_accuracy / cnt

            whole_size = sum([tf.size(_) for _ in new_variables_to_restore])
            print("Accuracy = {} with vocabulary {} inference used {} for model size {}".format(test_accuracy, vocabulary_size, inf_time, sess.run(whole_size)))
        
        sys.exit(1)

    if arg.compress and arg.variational:
        vocab = []
        metrics = []
        ratios = sess.run(test_model.embedding.embedding_logdropout_ratio).squeeze()
        ratios.sort()
        intervals = list(np.linspace(1, 100, 40)) + list(np.linspace(100, vocabulary_size - 1, 60))
        intervals = [ratios[int(_)] for _ in intervals]
        for t in intervals:
            data_processor_valid = DataProcessor(test_in_path, test_slot_path, test_intent_path, in_vocab, slot_vocab, intent_vocab)
            sum_accuracy, cnt = 0, 0
            while True:
                in_data, slot_data, slot_weight, length, intents, in_seq, slot_seq, intent_seq = data_processor_valid.get_batch(arg.batch_size)
                train_feed_dict = {
                    test_model.x: in_data,
                    test_model.y: intents,
                    test_model.sequence_length: length,
                    test_model.threshold: t
                }
                _, accuracy = sess.run([test_model.predictions, test_model.accuracy], feed_dict=train_feed_dict) 
                sum_accuracy += accuracy
                cnt += 1
                if data_processor_valid.end == 1:
                    break

            test_accuracy = sum_accuracy / cnt
            sparsity = sess.run(test_model.sparsity, feed_dict={test_model.threshold:t})
            rest_words = int((1 - sparsity) * vocabulary_size)
            if rest_words < 1:
                rest_words = 1
            metrics.append(test_accuracy)
            vocab.append(rest_words)
            data_processor_valid.close() 

        print metrics
        print vocab
        print("ROC={} CR={}".format(ROC(metrics, vocab, FULL_VOCAB), CR(metrics, vocab)))
        sys.exit(1)
    elif arg.compress and arg.bound:
        vocab = []
        metrics = []
        intervals = list(np.linspace(1, 100, 40)) + list(np.linspace(100, vocabulary_size, 60))
        for t in intervals:
            t = int(t)
            zeros = np.zeros((vocabulary_size, 1), 'float32')
            for _ in range(20):
                t = np.random.choice(range(vocabulary_size), t)
                zeros[t, :] = 1
                sum_accuracy, cnt = 0, 0
                data_processor_valid = DataProcessor(test_in_path, test_slot_path, test_intent_path, in_vocab, slot_vocab, intent_vocab)
                sum_accuracy, cnt = 0, 0
                while True:
                    in_data, slot_data, slot_weight, length, intents, in_seq, slot_seq, intent_seq = data_processor_valid.get_batch(arg.batch_size)
                    train_feed_dict = {
                        test_model.x: in_data,
                        test_model.y: intents,
                        test_model.sequence_length: length,
                        test_model.mask: zeros
                    }
                    _, accuracy = sess.run([test_model.predictions, test_model.accuracy], feed_dict=train_feed_dict) 
                    sum_accuracy += accuracy
                    cnt += 1
                    if data_processor_valid.end == 1:
                        break

                test_accuracy = sum_accuracy / cnt
                metrics.append(test_accuracy)
                vocab.append(t)
                data_processor_valid.close()
        print(metrics)
        print(vocab)
        sys.exit(1)
    elif arg.compress and arg.tf_idf:
        vocab = []
        metrics = []
        intervals = list(np.linspace(1, 100, 40)) + list(np.linspace(100, vocabulary_size - 1, 60))
        stats = pickle.load(open('vocab/{}_tf_idf.pickle'.format(arg.dataset)))
        tf_idf_neg_scores = [stats[i]['score'] if i in stats else 0 for i in range(vocabulary_size)] 
        for t in intervals:
            t = int(t)
            idxs = np.argpartition(tf_idf_neg_scores, t)[:t]
            zeros = np.zeros((vocabulary_size, 1), 'float32')
            zeros[idxs, :] = 1
            sum_accuracy, cnt = 0, 0
            data_processor_valid = DataProcessor(test_in_path, test_slot_path, test_intent_path, in_vocab, slot_vocab, intent_vocab)
            sum_accuracy, cnt = 0, 0
            while True:
                in_data, slot_data, slot_weight, length, intents, in_seq, slot_seq, intent_seq = data_processor_valid.get_batch(arg.batch_size)
                train_feed_dict = {
                    test_model.x: in_data,
                    test_model.y: intents,
                    test_model.sequence_length: length,
                    test_model.mask: zeros
                }
                _, accuracy = sess.run([test_model.predictions, test_model.accuracy], feed_dict=train_feed_dict) 
                sum_accuracy += accuracy
                cnt += 1
                if data_processor_valid.end == 1:
                    break

            test_accuracy = sum_accuracy / cnt
            metrics.append(test_accuracy)
            vocab.append(t)
            data_processor_valid.close() 
        print metrics
        print vocab
        print("ROC={} CR={}".format(ROC(metrics, vocab, FULL_VOCAB), CR(metrics, vocab)))
        sys.exit(1)
    elif arg.compress:
        vocab = []
        metrics = []
        intervals = list(np.linspace(1, 100, 40)) + list(np.linspace(100, vocabulary_size, 60))
        for t in intervals:
            t = int(t)
            zeros = np.zeros((vocabulary_size, 1), 'float32')
            zeros[:t, :] = 1

            sum_accuracy, cnt = 0, 0
            data_processor_valid = DataProcessor(test_in_path, test_slot_path, test_intent_path, in_vocab, slot_vocab, intent_vocab)
            sum_accuracy, cnt = 0, 0
            while True:
                in_data, slot_data, slot_weight, length, intents, in_seq, slot_seq, intent_seq = data_processor_valid.get_batch(arg.batch_size)
                train_feed_dict = {
                    test_model.x: in_data,
                    test_model.y: intents,
                    test_model.sequence_length: length,
                    test_model.mask: zeros
                }
                _, accuracy = sess.run([test_model.predictions, test_model.accuracy], feed_dict=train_feed_dict) 
                sum_accuracy += accuracy
                cnt += 1
                if data_processor_valid.end == 1:
                    break

            test_accuracy = sum_accuracy / cnt
            metrics.append(test_accuracy)
            vocab.append(t)
            data_processor_valid.close() 
        print metrics
        print vocab
        print("ROC={} CR={}".format(ROC(metrics, vocab, FULL_VOCAB), CR(metrics, vocab)))
        sys.exit(1)
    
    epochs = 0
    loss = 0.0
    data_processor = None
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
        cur_decay = 0.0001
        learning_rate = 1e-2
        train_feed_dict = {
            model.x: in_data,
            model.y: intents,
            model.sequence_length: length,
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
            data_processor.close()
            data_processor = None
            epochs += 1
            #valid(os.path.join(full_valid_path, arg.input_file), \
            #      os.path.join(full_valid_path, arg.slot_file), \
            #      os.path.join(full_valid_path, arg.intent_file))
            valid(test_in_path, test_slot_path, test_intent_path)
            #save_path = os.path.join(arg.model_path,'_step_' + str(step) + '_epochs_' + str(epochs) + '.ckpt')
            saver.save(sess, model_name)
            if epochs == arg.max_epochs:
                break
