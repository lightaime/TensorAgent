import tensorflow as tf

class Model(object):
    def __init__(self, state_dim, action_dim, entropy_beta=1e-3, optimizer=None, learning_rate=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.entropy_beta = entropy_beta
        self.learning_rate = learning_rate
        tf.reset_default_graph()
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = optimizer or tf.train.RMSPropOptimizer(self.learning_rate)

        with tf.name_scope("model_input"):
            self.input_state = tf.placeholder(tf.float32, shape=[None, self.state_dim], name="states")

        with tf.name_scope("model_target"):
            self.taken_actions = tf.placeholder(tf.int32, shape=[None, 1], name="taken_actions")
            self.future_rewards = tf.placeholder(tf.float32, shape=[None, 1], name="future_rewards")

        with tf.name_scope("model"):
            self.__create_model()

    def __create_policy_network(self):
        with tf.variable_scope("shared_network"):
            h1 = tf.layers.dense(self.input_state,
                                    units=32,
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name="hidden_1")

            h2 = tf.layers.dense(h1,
                                    units=32,
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name="hidden_2")

        with tf.variable_scope("policy_network"):
            self.policy_outputs = tf.layers.dense(h2,
                                    units=self.action_dim,
                                    activation=tf.nn.softmax,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name="policy_outputs")

    def __creat_base_line_network(self):
        with tf.variable_scope("shared_network"):
            h1 = tf.layers.dense(self.input_state,
                                    units=32,
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name="hidden_1",
                                    reuse=True)

            h2 = tf.layers.dense(h1,
                                    units=32,
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name="hidden_2",
                                    reuse=True)

        with tf.variable_scope("base_line_network"):
            self.base_line_outputs = tf.layers.dense(h2,
                                    units=1,
                                    activation=None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name="base_line_outputs")

    def __create_loss(self):
        with tf.name_scope("compute_policy_gradients"):
            self.log_probs = tf.log(tf.clip_by_value(self.policy_outputs, 1e-20, 1.0))

            # entropy loss of exploration
            self.entropy_loss = -tf.reduce_sum(self.policy_outputs * self.log_probs, reduction_indices=1)
            self.policy_loss = -tf.reduce_sum(tf.multiply(self.log_probs, tf.squeeze(tf.cast(tf.one_hot(self.taken_actions, self.action_dim), tf.float32))) * \
                                              (self.future_rewards - tf.stop_gradient(self.base_line_outputs)), reduction_indices=1)
            # self.policy_loss = -tf.reduce_sum(tf.multiply(self.log_probs, tf.squeeze(tf.cast(tf.one_hot(self.taken_actions, self.action_dim), tf.float32))) * \
                                              # (self.future_rewards), reduction_indices=1)
            td = self.base_line_outputs - self.future_rewards
            td = tf.clip_by_value(td, -5.0, 5.0)
            # self.base_line_loss = tf.nn.l2_loss(self.base_line_outputs - self.future_rewards)
            self.base_line_loss = tf.reduce_sum(td**2 / 2, reduction_indices=1)
            self.total_loss = tf.add_n([self.policy_loss,
                                      self.entropy_beta * self.entropy_loss,
                                      self.base_line_loss])

    def __create_train_policy_op(self):
        self.train_policy_op = self.optimizer.minimize(self.policy_loss,
                                                global_step=tf.contrib.framework.get_global_step())

    def __create_train_op(self):
        self.train_op = self.optimizer.minimize(self.total_loss,
                                                global_step=tf.contrib.framework.get_global_step())

    def __create_model(self):
        self.__create_policy_network()
        self.__creat_base_line_network()
        self.__create_loss()
        self.__create_train_policy_op()
        self.__create_train_op()
        self.__create_get_layer_weight_op()

    def predict_policy(self, feed_state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.policy_outputs, {self.input_state: feed_state})

    def update_policy(self, feed_state, feed_taken_actions, feed_future_rewards, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run([self. train_policy_op, self.policy_loss],
                        {self.input_state: feed_state,
                            self.taken_actions: feed_taken_actions,
                            self.future_rewards: feed_future_rewards})

    def predict_baseline(self, feed_state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.base_line_outputs, {self.input_state: feed_state})

    def update_baseline(self, feed_state, feed_future_rewards, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.base_line_loss, {self.input_state: feed_state,
                                              self.future_rewards: feed_future_rewards})

    def update(self, feed_state, feed_taken_actions, feed_future_rewards, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run([self.train_op, self.total_loss, self.policy_loss, self.base_line_loss],
                        {self.input_state: feed_state,
                            self.taken_actions: feed_taken_actions,
                            self.future_rewards: feed_future_rewards})

    def __create_get_layer_weight_op(self):
        with tf.name_scope("model"):
            with tf.variable_scope("shared_network", reuse=True):
                self.h1_weiget = tf.get_variable("hidden_1/kernel")
                self.h1_bias = tf.get_variable("hidden_1/bias")

    def run_layer_weight(self, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run([self.h1_weiget, self.h1_bias])


if __name__ == '__main__':
    model = Model(40, 5)
