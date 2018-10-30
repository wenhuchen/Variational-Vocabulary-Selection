import tensorflow as tf


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
