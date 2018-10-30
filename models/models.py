import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

class VarDropoutEmbedding(object):
    def __init__(self, input_size, layer_size, batch_size, name="embedding"):
        self.name = name
        self.input_size = input_size
        self.layer_size = layer_size
        self.batch_size = batch_size
        self.logdropout_init = tf.random_uniform_initializer(0, 3)        
        self.embedding_mean = tf.get_variable(name, [self.input_size, self.layer_size])
        self.embedding_logdropout_ratio = tf.get_variable(name + "_ratio", [self.input_size, 1], initializer=self.logdropout_init)
        self.eps = tf.random_normal([self.batch_size, 1, self.layer_size], 0.0, 1.0)      

    def __call__(self, input_data, sample=False, mask=None):
        if sample:
            output_mean = tf.nn.embedding_lookup(self.embedding_mean, input_data)
            output_logdropout = tf.nn.embedding_lookup(self.clip(self.embedding_logdropout_ratio), input_data)
            output_std = tf.exp(0.5 * output_logdropout) * output_mean
            output = output_mean + output_std * self.eps
        elif mask is None:
            output = tf.nn.embedding_lookup(self.clip(self.embedding_mean), input_data)
        else:
            output = tf.nn.embedding_lookup(mask * self.clip(self.embedding_mean), input_data)

        return output

    def clip(self, mtx, to=10):
        return tf.clip_by_value(mtx, -to, to)

    def zeroed_embedding(self):
        return self.embedding_mean * self.mask
    
    def l1_norm(self):
        t = tf.square(self.embedding_mean)
        t = tf.reduce_sum(t, axis=-1) + tf.constant(1.0e-8)
        t = tf.sqrt(t)
        reg = tf.reduce_sum(t)
        return reg

    def rowwise_norm(self):
        t = tf.square(self.embedding_mean)
        t = tf.reduce_sum(t, axis=-1) + tf.constant(1.0e-8)
        t = tf.sqrt(t)
        return t

    def regularizer(self):
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        log_alpha = self.clip(self.embedding_logdropout_ratio)
        KLD = -tf.reduce_sum(k1 * tf.sigmoid(k2 + k3 * log_alpha) - 0.5 * tf.nn.softplus(-log_alpha) - k1)
        return KLD    

class WordCNN(object):
    def __init__(self, vocabulary_size, document_max_len, num_class, emb_size, is_training, filter_sizes=[3, 4, 5], variational=False, l1=False, batch_size=128, compress=False):
        self.learning_rate = tf.placeholder(tf.float32, [], name="learning_rate")
        self.filter_sizes = filter_sizes
        self.num_filters = 100

        self.x = tf.placeholder(tf.int32, [None, document_max_len], name="x")
        self.y = tf.placeholder(tf.int32, [None], name="y")
        self.threshold = tf.placeholder(tf.float32, [], name='threshold')
        self.l1_threshold = tf.placeholder(tf.float32, [], name='l1_threshold')
        
        self.global_step = tf.Variable(0, trainable=False)

        if is_training:
            self.keep_prob = 0.5
        else:
            self.keep_prob = 1.0
        
        self.embedding = VarDropoutEmbedding(vocabulary_size, emb_size, batch_size)

        if variational:
            self.mask = tf.cast(tf.less(self.embedding.embedding_logdropout_ratio, self.threshold), tf.float32)
            self.sparsity = tf.nn.zero_fraction(self.mask)
        elif l1: 
            self.mask = tf.cast(tf.greater(tf.expand_dims(self.embedding.rowwise_norm(), -1), self.l1_threshold), tf.float32)
            self.sparsity = tf.nn.zero_fraction(self.mask) 
        else:
            self.mask = tf.placeholder(tf.float32, [vocabulary_size, 1], name="mask")
            self.sparsity = tf.constant(0.0, dtype=tf.float32)
        #if variational:
        #else:
        #    self.embedding = VarDropoutEmbedding(vocabulary_size, emb_size, batch_size, is_training=False)
        self.weight_decay = tf.placeholder(tf.float32, shape=(), name="weight_decay")

        if variational:
            if is_training:
                self.x_emb = tf.expand_dims(self.embedding(self.x, sample=True, mask=None), -1)
            else:
                self.x_emb = tf.expand_dims(self.embedding(self.x, sample=False, mask=self.mask), -1)
        elif l1:
            if is_training:
                self.x_emb = tf.expand_dims(self.embedding(self.x, sample=False, mask=None), -1)
            else:
                self.x_emb = tf.expand_dims(self.embedding(self.x, sample=False, mask=self.mask), -1)
        else:
            if not compress:
                self.x_emb = tf.expand_dims(self.embedding(self.x, sample=False, mask=None), -1)
            else:
                self.x_emb = tf.expand_dims(self.embedding(self.x, sample=False, mask=self.mask), -1)

        if variational:
            self.reg_loss = self.weight_decay * self.embedding.regularizer()
        else:
            self.reg_loss = tf.constant(0., dtype=tf.float32)
        if l1:
            self.reg_loss += self.weight_decay * self.embedding.l1_norm()
        else:
            self.reg_loss += tf.constant(0., dtype=tf.float32)
        
        trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        pooled_outputs = []
        for filter_size in self.filter_sizes:
            conv = tf.layers.conv2d(
                self.x_emb,
                filters=self.num_filters,
                kernel_size=[filter_size, emb_size],
                strides=(1, 1),
                padding="VALID",
                activation=tf.nn.relu)
            pool = tf.layers.max_pooling2d(
                conv,
                pool_size=[document_max_len - filter_size + 1, 1],
                strides=(1, 1),
                padding="VALID")
            pooled_outputs.append(pool)

        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters * len(self.filter_sizes)])

        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, self.keep_prob)

        with tf.name_scope("output"):
            self.logits = tf.layers.dense(h_drop, num_class, activation=None)
            self.predictions = tf.argmax(self.logits, -1, output_type=tf.int32)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        if is_training:
            with tf.name_scope("loss"):
                self.cross_entropy = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
                self.loss = self.cross_entropy + self.reg_loss
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step, var_list=trainables)


class WordRNN(object):
    def __init__(self, vocabulary_size, document_max_len, num_class, emb_size, is_training, variational=False, l1=False, batch_size=128, compress=False):
        self.learning_rate = tf.placeholder(tf.float32, [], name="learning_rate")

        self.num_hidden = num_hidden
        self.num_layers = 2
        
        self.x = tf.placeholder(tf.int32, [None, document_max_len], name="x")
        self.y = tf.placeholder(tf.int32, [None], name="y")
        self.threshold = tf.placeholder(tf.float32, [], name='threshold')
        self.l1_threshold = tf.placeholder(tf.float32, [], name='l1_threshold')
        
        self.global_step = tf.Variable(0, trainable=False)

        if is_training:
            self.keep_prob = 0.5
        else:
            self.keep_prob = 1.0
        
        self.embedding = VarDropoutEmbedding(vocabulary_size, emb_size, batch_size)

        if variational:
            self.mask = tf.cast(tf.less(self.embedding.embedding_logdropout_ratio, self.threshold), tf.float32)
            self.sparsity = tf.nn.zero_fraction(self.mask)
        elif l1: 
            self.mask = tf.cast(tf.greater(tf.expand_dims(self.embedding.rowwise_norm(), -1), self.l1_threshold), tf.float32)
            self.sparsity = tf.nn.zero_fraction(self.mask) 
        else:
            self.mask = tf.placeholder(tf.float32, [vocabulary_size, 1], name="mask")
            self.sparsity = tf.constant(0.0, dtype=tf.float32)

        if variational:
            if is_training:
                self.x_emb = tf.expand_dims(self.embedding(self.x, sample=True, mask=None), -1)
            else:
                self.x_emb = tf.expand_dims(self.embedding(self.x, sample=False, mask=self.mask), -1)
        elif l1:
            if is_training:
                self.x_emb = tf.expand_dims(self.embedding(self.x, sample=False, mask=None), -1)
            else:
                self.x_emb = tf.expand_dims(self.embedding(self.x, sample=False, mask=self.mask), -1)
        else:
            if not compress:
                self.x_emb = tf.expand_dims(self.embedding(self.x, sample=False, mask=None), -1)
            else:
                self.x_emb = tf.expand_dims(self.embedding(self.x, sample=False, mask=self.mask), -1)

        if variational:
            self.reg_loss = self.weight_decay * self.embedding.regularizer()
        else:
            self.reg_loss = tf.constant(0., dtype=tf.float32)
        if l1:
            self.reg_loss += self.weight_decay * self.embedding.l1_norm()
        else:
            self.reg_loss += tf.constant(0., dtype=tf.float32)

        with tf.name_scope("birnn"):
            fw_cells = [rnn.BasicLSTMCell(self.num_hidden) for _ in range(self.num_layers)]
            bw_cells = [rnn.BasicLSTMCell(self.num_hidden) for _ in range(self.num_layers)]
            fw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob) for cell in fw_cells]
            bw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob) for cell in bw_cells]

            rnn_outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(
                fw_cells, bw_cells, self.x_emb, sequence_length=self.x_len, dtype=tf.float32)
            rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, document_max_len * self.num_hidden * 2])

        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(rnn_outputs_flat, self.keep_prob)

        with tf.name_scope("output"):
            self.logits = tf.layers.dense(h_drop, num_class, activation=None)
            self.predictions = tf.argmax(self.logits, -1, output_type=tf.int32)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        if is_training:
            with tf.name_scope("loss"):
                self.cross_entropy = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
                self.loss = self.cross_entropy + self.reg_loss
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step, var_list=trainables)


class CharCNN(object):
    def __init__(self, alphabet_size, document_max_len, num_class, num_filters, is_training=True):
        self.learning_rate = tf.placeholder(tf.float32, [], name="learning_rate")
        self.filter_sizes = [7, 7, 3, 3, 3, 3]
        self.kernel_initializer = tf.truncated_normal_initializer(stddev=0.05)
        self.num_filters = num_filters

        self.x = tf.placeholder(tf.int32, [None, document_max_len], name="x")
        self.y = tf.placeholder(tf.int32, [None], name="y")
        self.global_step = tf.Variable(0, trainable=False)
        self.threshold = tf.placeholder(tf.float32, [], name='threshold')
        self.l1_threshold = tf.placeholder(tf.float32, [], name='l1_threshold')
        self.weight_decay = tf.placeholder(tf.float32, shape=(), name="weight_decay")

        if is_training:
            self.keep_prob = 0.5
        else:
            self.keep_prob = 1.0

        self.x_one_hot = tf.one_hot(self.x, alphabet_size)
        self.x_expanded = tf.expand_dims(self.x_one_hot, -1)

        # ============= Convolutional Layers =============
        with tf.name_scope("conv-maxpool-1"):
            conv1 = tf.layers.conv2d(
                self.x_expanded,
                filters=self.num_filters,
                kernel_size=[self.filter_sizes[0], alphabet_size],
                kernel_initializer=self.kernel_initializer,
                activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(
                conv1,
                pool_size=(3, 1),
                strides=(3, 1))
            pool1 = tf.transpose(pool1, [0, 1, 3, 2])

        with tf.name_scope("conv-maxpool-2"):
            conv2 = tf.layers.conv2d(
                pool1,
                filters=self.num_filters,
                kernel_size=[self.filter_sizes[1], self.num_filters],
                kernel_initializer=self.kernel_initializer,
                activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(
                conv2,
                pool_size=(3, 1),
                strides=(3, 1))
            pool2 = tf.transpose(pool2, [0, 1, 3, 2])

        with tf.name_scope("conv-3"):
            conv3 = tf.layers.conv2d(
                pool2,
                filters=self.num_filters,
                kernel_size=[self.filter_sizes[2], self.num_filters],
                kernel_initializer=self.kernel_initializer,
                activation=tf.nn.relu)
            conv3 = tf.transpose(conv3, [0, 1, 3, 2])

        with tf.name_scope("conv-4"):
            conv4 = tf.layers.conv2d(
                conv3,
                filters=self.num_filters,
                kernel_size=[self.filter_sizes[3], self.num_filters],
                kernel_initializer=self.kernel_initializer,
                activation=tf.nn.relu)
            conv4 = tf.transpose(conv4, [0, 1, 3, 2])

        with tf.name_scope("conv-5"):
            conv5 = tf.layers.conv2d(
                conv4,
                filters=self.num_filters,
                kernel_size=[self.filter_sizes[4], self.num_filters],
                kernel_initializer=self.kernel_initializer,
                activation=tf.nn.relu)
            conv5 = tf.transpose(conv5, [0, 1, 3, 2])

        with tf.name_scope("conv-maxpool-6"):
            conv6 = tf.layers.conv2d(
                conv5,
                filters=self.num_filters,
                kernel_size=[self.filter_sizes[5], self.num_filters],
                kernel_initializer=self.kernel_initializer,
                activation=tf.nn.relu)
            pool6 = tf.layers.max_pooling2d(
                conv6,
                pool_size=(3, 1),
                strides=(3, 1))
            pool6 = tf.transpose(pool6, [0, 2, 1, 3])
            h_pool = tf.reshape(pool6, [-1, 34 * self.num_filters])

        # ============= Fully Connected Layers =============
        with tf.name_scope("fc-1"):
            fc1_out = tf.layers.dense(h_pool, 1024, activation=tf.nn.relu, kernel_initializer=self.kernel_initializer)

        with tf.name_scope("fc-2"):
            fc2_out = tf.layers.dense(fc1_out, 1024, activation=tf.nn.relu, kernel_initializer=self.kernel_initializer)

        with tf.name_scope("fc-3"):
            self.logits = tf.layers.dense(fc2_out, num_class, activation=None, kernel_initializer=self.kernel_initializer)
            self.predictions = tf.argmax(self.logits, -1, output_type=tf.int32)
        
        self.sparsity = tf.constant(0.0, dtype=tf.float32)
        self.reg_loss = tf.constant(0., dtype=tf.float32)

        # ============= Loss and Accuracy =============
        with tf.name_scope("loss"):
            #self.y_one_hot = tf.one_hot(self.y, num_class)
            self.cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
            self.loss = self.cross_entropy + self.reg_loss
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

