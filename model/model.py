# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Model defintion
"""                                        

import tensorflow as tf        
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from utils import clone_variable_list, create_fc_layer, create_conv_layer

PARAM_XI_STEP = 1e-3
NEG_INF = -1e32
EPSILON = 1e-32
STOP_GRADIENTS = False

def weight_variable(shape, name='fc', init_type='default'):
    """
    Define weight variables
    Args:
        shape       Shape of the bias variable tensor

    Returns:
        A tensor of size shape initialized from a random normal
    """
    with tf.variable_scope(name):
        if init_type == 'default':
            weights = tf.get_variable('weights', shape, tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
            #weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weights')
        elif init_type == 'zero':
            weights = tf.get_variable('weights', shape, tf.float32, initializer=tf.constant_initializer(0.1))
            #weights = tf.Variable(tf.constant(0.1, shape=shape, dtype=np.float32), name='weights')

    return weights

def bias_variable(shape, name='fc'):
    """
    Define bias variables
    Args:
        shape       Shape of the bias variable tensor

    Returns:
        A tensor of size shape initialized from a constant
    """
    with tf.variable_scope(name):
        biases = tf.get_variable('biases', shape, initializer=tf.constant_initializer(0.1))

    return biases
    #return tf.Variable(tf.constant(0.1, shape=shape, dtype=np.float32), name='biases') #TODO: Should we initialize it from 0

class Model:
    """
    A class defining the model
    """

    def __init__(self, x_, y_, num_tasks, opt, imp_method, synap_stgth, fisher_update_after, fisher_ema_decay, model_learning_rate, 
                 network_arch='FC-S', is_ATT_DATASET=False, x_test=None, attr=None, anchor_xx=None, anchor_eta=0.1, anchor_points=None):
        """
        Instantiate the model
        """
        # Define some placeholders which are used to feed the data to the model
        self.x = x_
        self.y_ = y_
        self.total_classes = int(self.y_.get_shape()[1])
        self.learning_rate = model_learning_rate
        self.output_mask = tf.placeholder(dtype=tf.float32, shape=[self.total_classes])
        self.sample_weights = tf.placeholder(tf.float32, shape=[None])
        self.task_id = tf.placeholder(dtype=tf.int32, shape=())
        self.store_grad_batches = tf.placeholder(dtype=tf.float32, shape=())
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=())
        self.train_samples = tf.placeholder(dtype=tf.float32, shape=())
        self.training_iters = tf.placeholder(dtype=tf.float32, shape=())
        self.train_step = tf.placeholder(dtype=tf.float32, shape=())
        self.violation_count = tf.Variable(0, dtype=tf.float32, trainable=False)

        # Hindsight anchors
        self.anchor_xx = anchor_xx
        self.anchor_eta = anchor_eta
        self.anchor_points = anchor_points
        self.anchor_opt = tf.train.AdamOptimizer(learning_rate=0.001)

        # Save the arguments passed from the main script
        self.opt = opt
        self.num_tasks = num_tasks
        self.imp_method = imp_method
        self.fisher_update_after = fisher_update_after
        self.fisher_ema_decay = fisher_ema_decay
        self.network_arch = network_arch

        # A scalar variable for previous syanpse strength
        self.synap_stgth = tf.constant(synap_stgth, shape=[1], dtype=tf.float32)

        # Define different variables
        self.weights_old = []
        self.star_vars = []
        self.small_omega_vars = []
        self.big_omega_vars = []
        self.big_omega_riemann_vars = []
        self.fisher_diagonal_at_minima = []
        self.hebbian_score_vars = []
        self.running_fisher_vars = []
        self.tmp_fisher_vars = []
        self.max_fisher_vars = []
        self.min_fisher_vars = []
        self.max_score_vars = []
        self.min_score_vars = []
        self.normalized_score_vars = []
        self.score_vars = []
        self.normalized_fisher_at_minima_vars = []
        self.weights_delta_old_vars = []
        self.ref_grads = []
        self.projected_gradients_list = []
        self.theta_not_a = [] # MER
        self.theta_i_not_w = [] # MER
        self.theta_i_a = [] # MER

        self.loss_and_train_ops_for_one_hot_vector(self.x, self.y_)

        # Set the operations to reset the optimier when needed
        self.reset_optimizer_ops()
    
####################################################################################
#### Internal APIs of the class. These should not be called/ exposed externally ####
####################################################################################
    def loss_and_train_ops_for_one_hot_vector(self, x, y_):
        """
        Loss and training operations for the training of one-hot vector based classification model
        """
        # Define approproate network
        if self.network_arch == 'FC-S':
            input_dim = int(x.get_shape()[1])
            layer_dims = [input_dim, 256, 256, self.total_classes]
            self.fc_variables(layer_dims)
            logits = self.fc_feedforward(x, self.trainable_vars)

        elif self.network_arch == 'FC-B':
            input_dim = int(x.get_shape()[1])
            layer_dims = [input_dim, 2000, 2000, self.total_classes]
            self.fc_variables(layer_dims)
            logits = self.fc_feedforward(x, self.trainable_vars)


        # Define operations for computing the running average of the class specific features
        self.running_average_of_features()

        # Create list of variables for storing different measures
        # Note: This method has to be called before calculating fisher 
        # or any other importance measure
        self.init_vars()

        # Different entropy measures/ loss definitions
        self.mse = 2.0*tf.nn.l2_loss(logits) # tf.nn.l2_loss computes sum(T**2)/ 2
        self.weighted_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_, 
            logits, self.sample_weights, reduction=tf.losses.Reduction.NONE))
        self.unweighted_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, 
            logits=logits))

        # Create operations for loss and gradient calculation
        self.loss_and_gradients(self.imp_method)

        # Store the current weights before doing a train step
        self.get_current_weights()

        # For GEM variants train ops will be defined later
        if 'GEM' not in self.imp_method:
            # Define the training operation here as Pathint ops depend on the train ops
            self.train_op()

        # Create operations to compute importance depending on the importance methods
        if self.imp_method == 'EWC':
            self.create_fisher_ops()
        elif self.imp_method == 'M-EWC':
            self.create_fisher_ops()
            self.create_pathint_ops()
            self.combined_fisher_pathint_ops()
        elif self.imp_method == 'PI':
            self.create_pathint_ops()
        elif self.imp_method == 'RWALK':
            self.create_fisher_ops()
            self.create_pathint_ops()
        elif self.imp_method == 'MAS':
            self.create_hebbian_ops()
        elif self.imp_method == 'A-GEM' or self.imp_method == 'S-GEM':
            self.create_stochastic_gem_ops()
        elif self.imp_method == 'MER':
            self.mer_beta = tf.placeholder(dtype=tf.float32, shape=())
            self.mer_gamma = tf.placeholder(dtype=tf.float32, shape=())
            self.create_mer_ops()

        # Create weight save and store ops
        self.weights_store_ops()

        #  Summary operations for visualization
        tf.summary.scalar("unweighted_entropy", self.unweighted_entropy)
        for v in self.trainable_vars:
            tf.summary.histogram(v.name.replace(":", "_"), v)
        self.merged_summary = tf.summary.merge_all()

        self.correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32))
    
    
    def fc_variables(self, layer_dims):
        """
        Defines variables for a 3-layer fc network
        Args:

        Returns:
        """

        self.weights = []
        self.biases = []
        self.trainable_vars = []

        for i in range(len(layer_dims)-1):
            w = weight_variable([layer_dims[i], layer_dims[i+1]], name='fc_%d'%(i))
            b = bias_variable([layer_dims[i+1]], name='fc_%d'%(i))
            self.weights.append(w)
            self.biases.append(b)
            self.trainable_vars.append(w)
            self.trainable_vars.append(b)

    def fc_feedforward(self, h, weights, store_features=True, store_synthetic_features=False, apply_dropout=False):
        """
        Forward pass through a fc network
        Args:
            h               Input image (tensor)
            weights         List of weights and biases
            apply_dropout   Whether to apply droupout (True/ False)

        Returns:
            Logits of a fc network
        """
        num_layers = len(weights)// 2
        if apply_dropout:
            h = tf.nn.dropout(h, 1) # Apply dropout on Input?
        #for (w, b) in list(zip(weights, biases))[:-1]:
        for ii in range(num_layers-1): # Last layer weight and biases won't have a non-linearity
            offset = ii*2
            w = weights[offset]
            b = weights[offset+1]
            h = create_fc_layer(h, w, b)
            if apply_dropout:
                h = tf.nn.dropout(h, 1)  # Apply dropout on hidden layers?

        # Store image features
        if store_features:
            self.features = h
            self.image_feature_dim = h.get_shape().as_list()[-1]
            if store_synthetic_features:
                return h, create_fc_layer(h, weights[-2], weights[-1], apply_relu=False)
        
        return create_fc_layer(h, weights[-2], weights[-1], apply_relu=False)

    """
    def fc_feedforward(self, h, weights, biases, apply_dropout=False):
        if apply_dropout:
            h = tf.nn.dropout(h, 1) # Apply dropout on Input?
        for (w, b) in list(zip(weights, biases))[:-1]:
            h = create_fc_layer(h, w, b)
            if apply_dropout:
                h = tf.nn.dropout(h, 1)  # Apply dropout on hidden layers?

        # Store image features 
        self.features = h
        self.image_feature_dim = h.get_shape().as_list()[-1]
        return create_fc_layer(h, weights[-1], biases[-1], apply_relu=False)
    """

    def loss_and_gradients(self, imp_method):
        """
        Defines task based and surrogate losses and their
        gradients
        Args:

        Returns:
        """
        reg = 0.0
        if imp_method == 'VAN'  or 'ER-' in imp_method or 'GEM' in imp_method or imp_method == 'MER':
            pass
        elif imp_method == 'EWC' or imp_method == 'M-EWC':
            reg = tf.add_n([tf.reduce_sum(tf.square(w - w_star) * f) for w, w_star, 
                f in zip(self.trainable_vars, self.star_vars, self.normalized_fisher_at_minima_vars)])
        elif imp_method == 'PI':
            reg = tf.add_n([tf.reduce_sum(tf.square(w - w_star) * f) for w, w_star, 
                f in zip(self.trainable_vars, self.star_vars, self.big_omega_vars)])
        elif imp_method == 'MAS':
            reg = tf.add_n([tf.reduce_sum(tf.square(w - w_star) * f) for w, w_star, 
                f in zip(self.trainable_vars, self.star_vars, self.hebbian_score_vars)])
        elif imp_method == 'RWALK':
            reg = tf.add_n([tf.reduce_sum(tf.square(w - w_star) * (f + scr)) for w, w_star, 
                f, scr in zip(self.trainable_vars, self.star_vars, self.normalized_fisher_at_minima_vars, 
                    self.normalized_score_vars)])
     
        # Regularized training loss
        self.reg_loss = tf.squeeze(self.unweighted_entropy + self.synap_stgth * reg)
        # Compute the gradients of the vanilla loss
        self.vanilla_gradients_vars = self.opt.compute_gradients(self.unweighted_entropy, 
                var_list=self.trainable_vars)
        # Compute the gradients of regularized loss
        self.reg_gradients_vars = self.opt.compute_gradients(self.reg_loss, 
                    var_list=self.trainable_vars)
        
        if imp_method == 'ER-Hindsight-Anchors':
            self.create_hindsight_anchor_ops_FC()
            self.create_er_anchor_ops_FC()
        

    def train_op(self):
        """
        Defines the training operation (a single step during training)
        Args:

        Returns:
        """
        if self.imp_method == 'VAN' or 'ER-' in self.imp_method or self.imp_method == 'MER':
            # Define training operation
            self.train = self.opt.apply_gradients(self.reg_gradients_vars)
        elif self.imp_method == 'FTR_EXT':
            # Define a training operation for the first and subsequent tasks
            self.train = self.opt.apply_gradients(self.reg_gradients_vars)
            self.train_classifier = self.opt.apply_gradients(self.reg_gradients_vars[-2:])
        else:
            # Get the value of old weights first
            with tf.control_dependencies([self.weights_old_ops_grouped]):
                # Define a training operation
                self.train = self.opt.apply_gradients(self.reg_gradients_vars)

    def init_vars(self):
        """
        Defines different variables that will be used for the
        weight consolidation
        Args:

        Returns:
        """

        if self.imp_method == 'PNN':
            return

        for v in range(len(self.trainable_vars)):

            # List of variables for weight updates
            self.weights_old.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            self.weights_delta_old_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            self.star_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False, 
                                                  name=self.trainable_vars[v].name.rsplit(':')[0]+'_star'))

            # List of variables for pathint method
            self.small_omega_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            self.big_omega_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            self.big_omega_riemann_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))

            # List of variables to store fisher information
            self.fisher_diagonal_at_minima.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))

            self.normalized_fisher_at_minima_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False, dtype=tf.float32))
            self.tmp_fisher_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            self.running_fisher_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            self.score_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            # New variables for conv setting for fisher and score normalization
            self.max_fisher_vars.append(tf.Variable(tf.zeros(1), dtype=tf.float32, trainable=False))
            self.min_fisher_vars.append(tf.Variable(tf.zeros(1), dtype=tf.float32, trainable=False))
            self.max_score_vars.append(tf.Variable(tf.zeros(1), dtype=tf.float32, trainable=False))
            self.min_score_vars.append(tf.Variable(tf.zeros(1), dtype=tf.float32, trainable=False))
            self.normalized_score_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            if self.imp_method == 'MAS':
                # List of variables to store hebbian information
                self.hebbian_score_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            elif self.imp_method == 'A-GEM' or self.imp_method == 'S-GEM':
                self.ref_grads.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
                self.projected_gradients_list.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            elif 'MER' in self.imp_method:
                # Variables to store parameters \theta_0^A in the paper
                self.theta_not_a.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False,
                                                  name=self.trainable_vars[v].name.rsplit(':')[0]+'_theta_not_a'))
                # Variables to store parameters \theta_{i,0}^W in the paper
                self.theta_i_not_w.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False,
                                                  name=self.trainable_vars[v].name.rsplit(':')[0]+'_theta_i_not_w'))
                self.theta_i_a.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False,
                                                  name=self.trainable_vars[v].name.rsplit(':')[0]+'_theta_i_not_w'))

    def get_current_weights(self):
        """
        Get the values of current weights
        Note: These weights are different from star_vars as those
        store the weights after training for the last task.
        Args:

        Returns:
        """
        weights_old_ops = []
        weights_delta_old_ops = []
        for v in range(len(self.trainable_vars)):
            weights_old_ops.append(tf.assign(self.weights_old[v], self.trainable_vars[v]))
            weights_delta_old_ops.append(tf.assign(self.weights_delta_old_vars[v], self.trainable_vars[v]))

        self.weights_old_ops_grouped = tf.group(*weights_old_ops)
        self.weights_delta_old_grouped = tf.group(*weights_delta_old_ops)


    def weights_store_ops(self):
        """
        Defines weight restoration operations
        Args:

        Returns:
        """
        restore_weights_ops = []
        set_star_vars_ops = []

        for v in range(len(self.trainable_vars)):
            restore_weights_ops.append(tf.assign(self.trainable_vars[v], self.star_vars[v]))

            set_star_vars_ops.append(tf.assign(self.star_vars[v], self.trainable_vars[v]))

        self.restore_weights = tf.group(*restore_weights_ops)
        self.set_star_vars = tf.group(*set_star_vars_ops)

    def reset_optimizer_ops(self):
        """
        Defines operations to reset the optimizer
        Args:

        Returns:
        """
        # Set the operation for resetting the optimizer
        self.optimizer_slots = [self.opt.get_slot(var, name) for name in self.opt.get_slot_names()\
                           for var in tf.global_variables() if self.opt.get_slot(var, name) is not None]
        self.slot_names = self.opt.get_slot_names()
        self.opt_init_op = tf.variables_initializer(self.optimizer_slots)

    def create_pathint_ops(self):
        """
        Defines operations for path integral-based importance
        Args:

        Returns:
        """
        reset_small_omega_ops = []
        update_small_omega_ops = []
        update_big_omega_ops = []
        update_big_omega_riemann_ops = []

        for v in range(len(self.trainable_vars)):
            # Make sure that the variables are updated before calculating delta(theta)
            with tf.control_dependencies([self.train]):
                update_small_omega_ops.append(tf.assign_add(self.small_omega_vars[v], 
                    -(self.vanilla_gradients_vars[v][0] * (self.trainable_vars[v] - self.weights_old[v]))))

            # Ops to reset the small omega
            reset_small_omega_ops.append(tf.assign(self.small_omega_vars[v], self.small_omega_vars[v]*0.0))

            if self.imp_method == 'PI':
                # Update the big omegas at the end of the task using the Eucldeian distance
                update_big_omega_ops.append(tf.assign_add(self.big_omega_vars[v], 
                    tf.nn.relu(tf.div(self.small_omega_vars[v], (PARAM_XI_STEP + tf.square(self.trainable_vars[v] - self.star_vars[v]))))))
            elif self.imp_method == 'RWALK':
                # Update the big omegas after small intervals using distance in riemannian manifold (KL-divergence)
                update_big_omega_riemann_ops.append(tf.assign_add(self.big_omega_riemann_vars[v], 
                    tf.nn.relu(tf.div(self.small_omega_vars[v], 
                        (PARAM_XI_STEP + self.running_fisher_vars[v] * tf.square(self.trainable_vars[v] - self.weights_delta_old_vars[v]))))))


        self.update_small_omega = tf.group(*update_small_omega_ops)
        self.reset_small_omega = tf.group(*reset_small_omega_ops)
        if self.imp_method == 'PI':
            self.update_big_omega = tf.group(*update_big_omega_ops)
        elif self.imp_method == 'RWALK':
            self.update_big_omega_riemann = tf.group(*update_big_omega_riemann_ops)
            self.big_omega_riemann_reset = [tf.assign(tensor, tf.zeros_like(tensor)) for tensor in self.big_omega_riemann_vars]

        if self.imp_method == 'RWALK':
            # For the first task, scale the scores so that division does not have an effect        
            self.scale_score = [tf.assign(s, s*2.0) for s in self.big_omega_riemann_vars]
            # To reduce the rigidity after each task the importance scores are averaged
            self.update_score = [tf.assign_add(scr, tf.div(tf.add(scr, riemm_omega), 2.0)) 
                    for scr, riemm_omega in zip(self.score_vars, self.big_omega_riemann_vars)]

            # Get the min and max in each layer of the scores
            self.get_max_score_vars = [tf.assign(var, tf.expand_dims(tf.squeeze(tf.reduce_max(scr, keepdims=True)), 
                axis=0)) for var, scr in zip(self.max_score_vars, self.score_vars)]
            self.get_min_score_vars = [tf.assign(var, tf.expand_dims(tf.squeeze(tf.reduce_min(scr, keepdims=True)), 
                axis=0)) for var, scr in zip(self.min_score_vars, self.score_vars)]
            self.max_score = tf.reduce_max(tf.convert_to_tensor(self.max_score_vars))
            self.min_score = tf.reduce_min(tf.convert_to_tensor(self.min_score_vars))
            with tf.control_dependencies([self.max_score, self.min_score]):
                self.normalize_scores = [tf.assign(tgt, (var - self.min_score)/ (self.max_score - self.min_score + EPSILON)) 
                        for tgt, var in zip(self.normalized_score_vars, self.score_vars)]

            # Sparsify all the layers except last layer
            sparsify_score_ops = []
            for v in range(len(self.normalized_score_vars) - 2):
                sparsify_score_ops.append(tf.assign(self.normalized_score_vars[v], 
                    tf.nn.dropout(self.normalized_score_vars[v], self.keep_prob)))

            self.sparsify_scores = tf.group(*sparsify_score_ops)

    def create_fisher_ops(self):
        """
        Defines the operations to compute online update of Fisher
        Args:

        Returns:
        """
        ders = tf.gradients(self.unweighted_entropy, self.trainable_vars)
        fisher_ema_at_step_ops = []
        fisher_accumulate_at_step_ops = []

        # ops for running fisher
        self.set_tmp_fisher = [tf.assign_add(f, tf.square(d)) for f, d in zip(self.tmp_fisher_vars, ders)]

        # Initialize the running fisher to non-zero value
        self.set_initial_running_fisher = [tf.assign(r_f, s_f) for r_f, s_f in zip(self.running_fisher_vars,
                                                                           self.tmp_fisher_vars)]

        self.set_running_fisher = [tf.assign(f, (1 - self.fisher_ema_decay) * f + (1.0/ self.fisher_update_after) *
                                    self.fisher_ema_decay * tmp) for f, tmp in zip(self.running_fisher_vars, self.tmp_fisher_vars)]

        self.get_fisher_at_minima = [tf.assign(var, f) for var, f in zip(self.fisher_diagonal_at_minima,
                                                                         self.running_fisher_vars)]

        self.reset_tmp_fisher = [tf.assign(tensor, tf.zeros_like(tensor)) for tensor in self.tmp_fisher_vars]

        # Get the min and max in each layer of the Fisher
        self.get_max_fisher_vars = [tf.assign(var, tf.expand_dims(tf.squeeze(tf.reduce_max(scr, keepdims=True)), axis=0)) 
                for var, scr in zip(self.max_fisher_vars, self.fisher_diagonal_at_minima)]
        self.get_min_fisher_vars = [tf.assign(var, tf.expand_dims(tf.squeeze(tf.reduce_min(scr, keepdims=True)), axis=0)) 
                for var, scr in zip(self.min_fisher_vars, self.fisher_diagonal_at_minima)]
        self.max_fisher = tf.reduce_max(tf.convert_to_tensor(self.max_fisher_vars))
        self.min_fisher = tf.reduce_min(tf.convert_to_tensor(self.min_fisher_vars))
        with tf.control_dependencies([self.max_fisher, self.min_fisher]):
            self.normalize_fisher_at_minima = [tf.assign(tgt, 
                (var - self.min_fisher)/ (self.max_fisher - self.min_fisher + EPSILON)) 
                    for tgt, var in zip(self.normalized_fisher_at_minima_vars, self.fisher_diagonal_at_minima)]

        self.clear_attr_embed_reg = tf.assign(self.normalized_fisher_at_minima_vars[-2], tf.zeros_like(self.normalized_fisher_at_minima_vars[-2]))

        # Sparsify all the layers except last layer
        sparsify_fisher_ops = []
        for v in range(len(self.normalized_fisher_at_minima_vars) - 2):
            sparsify_fisher_ops.append(tf.assign(self.normalized_fisher_at_minima_vars[v],
                tf.nn.dropout(self.normalized_fisher_at_minima_vars[v], self.keep_prob)))

        self.sparsify_fisher = tf.group(*sparsify_fisher_ops)

    def combined_fisher_pathint_ops(self):
        """
        Define the operations to refine Fisher information based on parameters convergence
        Args:

        Returns:
        """
        #self.refine_fisher_at_minima = [tf.assign(f, f*(1.0/(s+1e-12))) for f, s in zip(self.fisher_diagonal_at_minima, self.small_omega_vars)]
        self.refine_fisher_at_minima = [tf.assign(f, f*tf.exp(-100.0*s)) for f, s in zip(self.fisher_diagonal_at_minima, self.small_omega_vars)]


    def create_hebbian_ops(self):
        """
        Define operations for hebbian measure of importance (MAS)
        """
        # Compute the gradients of mse loss
        self.mse_gradients = tf.gradients(self.mse, self.trainable_vars)
        #with tf.control_dependencies([self.mse_gradients]):
        # Keep on adding gradients to the omega
        self.accumulate_hebbian_scores = [tf.assign_add(omega, tf.abs(grad)) for omega, grad in zip(self.hebbian_score_vars, self.mse_gradients)]
        # Average across the total images
        self.average_hebbian_scores = [tf.assign(omega, omega*(1.0/self.train_samples)) for omega in self.hebbian_score_vars]
        # Reset the hebbian importance variables
        self.reset_hebbian_scores = [tf.assign(omega, tf.zeros_like(omega)) for omega in self.hebbian_score_vars]

    def create_stochastic_gem_ops(self):
        """
        Define operations for Stochastic GEM
        """
        self.agem_loss = self.unweighted_entropy

        ref_grads = tf.gradients(self.agem_loss, self.trainable_vars)
        # Reference gradient for previous tasks
        self.store_ref_grads = [tf.assign(ref, grad) for ref, grad in zip(self.ref_grads, ref_grads)]
        flat_ref_grads =  tf.concat([tf.reshape(grad, [-1]) for grad in self.ref_grads], 0)
        # Grandient on the current task
        task_grads = tf.gradients(self.agem_loss, self.trainable_vars)
        flat_task_grads = tf.concat([tf.reshape(grad, [-1]) for grad in task_grads], 0)
        with tf.control_dependencies([flat_task_grads]):
            dotp = tf.reduce_sum(tf.multiply(flat_task_grads, flat_ref_grads))
            ref_mag = tf.reduce_sum(tf.multiply(flat_ref_grads, flat_ref_grads))
            proj = flat_task_grads - ((dotp/ ref_mag) * flat_ref_grads)
            self.reset_violation_count = self.violation_count.assign(0)
            def increment_violation_count():
                with tf.control_dependencies([tf.assign_add(self.violation_count, 1)]):
                    return tf.identity(self.violation_count)
            self.violation_count = tf.cond(tf.greater_equal(dotp, 0), lambda: tf.identity(self.violation_count), increment_violation_count)
            projected_gradients = tf.cond(tf.greater_equal(dotp, 0), lambda: tf.identity(flat_task_grads), lambda: tf.identity(proj))
            # Convert the flat projected gradient vector into a list
            offset = 0
            store_proj_grad_ops = []
            for v in self.projected_gradients_list:
                shape = v.get_shape()
                v_params = 1
                for dim in shape:
                    v_params *= dim.value
                store_proj_grad_ops.append(tf.assign(v, tf.reshape(projected_gradients[offset:offset+v_params], shape)))
                offset += v_params
            self.store_proj_grads = tf.group(*store_proj_grad_ops)
            # Define training operations for the tasks > 1
            with tf.control_dependencies([self.store_proj_grads]):
                self.train_subseq_tasks = self.opt.apply_gradients(zip(self.projected_gradients_list, self.trainable_vars))

        # Define training operations for the first task
        self.first_task_gradients_vars = self.opt.compute_gradients(self.agem_loss, var_list=self.trainable_vars)
        self.train_first_task = self.opt.apply_gradients(self.first_task_gradients_vars)
       
    def create_mer_ops(self):
        """
        Define operations for Meta-Experience replay
        """
        # Operation to store \theta_0^A
        self.store_theta_not_a = [tf.assign(var, val) for var, val in zip(self.theta_not_a, self.trainable_vars)]
        # Operation to store \theta_{i,0}^W
        self.store_theta_i_not_w = [tf.assign(var, val) for var, val in zip(self.theta_i_not_w, self.trainable_vars)]
        # Operation to store \theta_i^W
        self.store_theta_i_a = [tf.assign(var, val) for var, val in zip(self.theta_i_a, self.trainable_vars)]
        # With in batch reptile update
        self.with_in_batch_reptile_update = [tf.assign(var, val + self.mer_beta * (var - val)) for var, val in zip(self.trainable_vars, self.theta_i_not_w)]
        # Across the batch reptile update
        self.across_batch_reptile_update = [tf.assign(var, val1 + self.mer_gamma * (val2 - val1)) for var, val1, val2 in zip(self.trainable_vars, self.theta_not_a, self.theta_i_a)]

    def create_hindsight_anchor_ops_FC(self):
        """
        Hindsight Anchor operations for FC Network
        """
        def compute_forgetting_loss(x, star_vars, train_vars):
            synthetic_features, logits_at_theta_star = self.fc_feedforward(x, star_vars, store_synthetic_features=True)
            loss_at_theta_star = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_,
                                    logits=logits_at_theta_star))
            logits_at_theta = self.fc_feedforward(x, train_vars, store_features=False)
            loss_at_theta = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_,
                                    logits=logits_at_theta))
            negForgetting_loss = loss_at_theta_star - loss_at_theta
            return negForgetting_loss, synthetic_features

        self.negForgetting_loss, synthetic_features = compute_forgetting_loss(self.anchor_xx, self.star_vars, self.trainable_vars)
        self.phi_distance = tf.losses.cosine_distance(tf.math.l2_normalize(self.phi_hat_reference),
                                                      tf.math.l2_normalize(tf.squeeze(synthetic_features)), axis=0, reduction='weighted_mean')
        self.hindsight_objective = self.negForgetting_loss + self.anchor_eta*self.phi_distance

        hindsight_anchor_grad_vars = self.anchor_opt.compute_gradients(self.hindsight_objective, var_list=self.anchor_xx)
        self.update_hindsight_anchor = self.anchor_opt.apply_gradients(hindsight_anchor_grad_vars)
        self.reset_anchor_xx = tf.initialize_variables([self.anchor_xx])
    
    def create_er_anchor_ops_FC(self):
        """
        Anchoring objective operations for FC
        """
        # Temporary gradient update on the batch from the current task and replay buffer
        tmp_param_grads = tf.gradients(self.unweighted_entropy, self.trainable_vars)
        if STOP_GRADIENTS:  # First order approximation
            tmp_param_grads = [tf.stop_gradient(grad) for grad in tmp_param_grads]
        tmp_params = [w - self.learning_rate*g for w, g in zip(self.trainable_vars, tmp_param_grads)]

        # Check how much the updated parameters change the function values at the anchoring points
        anchor_logits = self.fc_feedforward(self.anchor_points, self.trainable_vars, store_features=False)
        tmp_logits = self.fc_feedforward(self.anchor_points, tmp_params, store_features=False)
        task_logits = self.fc_feedforward(self.x, self.trainable_vars)
        self.anchor_loss = tf.reduce_mean(tf.reduce_sum((anchor_logits-tmp_logits)**2, axis=1))
        self.task_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_,
                                        logits=task_logits))
        self.final_anchor_loss = tf.squeeze(self.task_loss + self.synap_stgth*self.anchor_loss)
        self.anchor_gradients_vars = self.opt.compute_gradients(self.final_anchor_loss,
                    var_list=self.trainable_vars)
        self.train_anchor = self.opt.apply_gradients(self.anchor_gradients_vars)

    def running_average_of_features(self):
        """
        Define operations for the running average of features
        """
        ####### \hat{phi}_i = \alpha * \hat{phi}_i + (1 - \alpha) * phi_i ------> Eq.1 ##############
        # Placeholder for controlling the running average weight
        self.phi_hat_alpha = tf.placeholder(dtype=tf.float32, shape=())
        # Placeholder for phi_hat to be used as groundtruth later on for the hindsight ER
        self.phi_hat_reference = tf.placeholder(tf.float32, shape=[self.image_feature_dim])
        # Variable for storing class specific features average
        self.phi_hat = tf.get_variable('mean_activations_per_class', [self.total_classes, self.image_feature_dim], tf.float32, initializer=tf.constant_initializer(0.0), trainable=False)
        # Mask for the average features
        phi_hat_mask = tf.Variable(tf.ones(self.phi_hat.get_shape(), dtype=tf.float32), trainable=False)
        # Get the class indices present in the batch
        class_indices = tf.where(tf.not_equal(self.y_, tf.constant(0, dtype=tf.float32)))[:, 1]
        # Update the mask (* alpha) for the classes present in the batch. If there are multiple examples from the same class in the batch
        # only one of the example will set the mask (property of scatter_update)
        phi_hat_mask = tf.scatter_update(phi_hat_mask, class_indices, self.phi_hat_alpha*tf.ones_like(self.features))
        # Multiply the existing value of running sum with \alpha => (\alpha * \hat{phi}_i of Eq. 1 above)
        self.phi_hat = tf.scatter_update(self.phi_hat, class_indices, tf.multiply(tf.gather(phi_hat_mask, class_indices), tf.gather(self.phi_hat, class_indices)))
        # Add the (1 - \alpha) * phi in the running sum
        self.phi_hat = tf.scatter_add(self.phi_hat, class_indices, (1-self.phi_hat_alpha)*self.features)
        # Reset op for the h_hat
        #self.reset_phi_hat = tf.initialize_variables([self.phi_hat])
        self.reset_phi_hat = tf.assign(self.phi_hat, tf.zeros_like(self.phi_hat))
#################################################################################
#### External APIs of the class. These will be called/ exposed externally #######
#################################################################################
    def reset_optimizer(self, sess):
        """
        Resets the optimizer state
        Args:
            sess        TF session

        Returns:
        """

        # Call the reset optimizer op
        sess.run(self.opt_init_op)

    def set_active_outputs(self, sess, labels):
        """
        Set the mask for the labels seen so far
        Args:
            sess        TF session
            labels      Mask labels

        Returns:
        """
        new_mask = np.zeros(self.total_classes)
        new_mask[labels] = 1.0
        """
        for l in labels:
            new_mask[l] = 1.0
        """
        sess.run(self.output_mask.assign(new_mask))

    def init_updates(self, sess):
        """
        Initialization updates
        Args:
            sess        TF session

        Returns:
        """
        # Set the star values to the initial weights, so that we can calculate
        # big_omegas reliably
        sess.run(self.set_star_vars)

    def task_updates(self, sess, task, train_x, train_labels, num_classes_per_task=10, online_cross_val=False):
        """
        Updates different variables when a task is completed
        Args:
            sess                TF session
            task                Task ID
            train_x             Training images for the task 
            train_labels        Labels in the task
        Returns:
        """
        if self.imp_method == 'VAN' or self.imp_method == 'PNN':
            # We'll store the current parameters at the end of this function
            pass
        elif self.imp_method == 'EWC':
            # Get the fisher at the end of a task
            sess.run(self.get_fisher_at_minima)
            # Normalize the fisher
            sess.run([self.get_max_fisher_vars, self.get_min_fisher_vars])
            sess.run([self.min_fisher, self.max_fisher, self.normalize_fisher_at_minima])
            # Don't regularize over the attribute-embedding vectors
            #sess.run(self.clear_attr_embed_reg)
            # Reset the tmp fisher vars
            sess.run(self.reset_tmp_fisher)
        elif self.imp_method == 'M-EWC':
            # Get the fisher at the end of a task
            sess.run(self.get_fisher_at_minima)
            # Refine Fisher based on the convergence info
            sess.run(self.refine_fisher_at_minima)
            # Normalize the fisher
            sess.run([self.get_max_fisher_vars, self.get_min_fisher_vars])
            sess.run([self.min_fisher, self.max_fisher, self.normalize_fisher_at_minima])
            # Reset the tmp fisher vars
            sess.run(self.reset_tmp_fisher)
            # Reset the small_omega_vars
            sess.run(self.reset_small_omega)
        elif self.imp_method == 'PI':
            # Update big omega variables
            sess.run(self.update_big_omega)
            # Reset the small_omega_vars because big_omega_vars are updated before it
            sess.run(self.reset_small_omega)
        elif self.imp_method == 'RWALK':
            if task == 0:
                # If first task then scale by a factor of 2, so that subsequent averaging does not hurt
                sess.run(self.scale_score)
            # Get the updated importance score
            sess.run(self.update_score)
            # Normalize the scores 
            sess.run([self.get_max_score_vars, self.get_min_score_vars])
            sess.run([self.min_score, self.max_score, self.normalize_scores])
            # Sparsify scores
            """
            # TODO: Tmp remove this?
            kp = 0.8 + (task*0.5)
            if (kp > 1):
                kp = 1.0
            """
            #sess.run(self.sparsify_scores, feed_dict={self.keep_prob: kp})
            # Get the fisher at the end of a task
            sess.run(self.get_fisher_at_minima)
            # Normalize fisher
            sess.run([self.get_max_fisher_vars, self.get_min_fisher_vars])
            sess.run([self.min_fisher, self.max_fisher, self.normalize_fisher_at_minima])
            # Sparsify fisher
            #sess.run(self.sparsify_fisher, feed_dict={self.keep_prob: kp})
            # Store the weights
            sess.run(self.weights_delta_old_grouped)
            # Reset the small_omega_vars because big_omega_vars are updated before it
            sess.run(self.reset_small_omega)
            # Reset the big_omega_riemann because importance score is stored in the scores array
            sess.run(self.big_omega_riemann_reset)
            # Reset the tmp fisher vars
            sess.run(self.reset_tmp_fisher)
        elif self.imp_method == 'MAS':
            # zero out any previous values
            sess.run(self.reset_hebbian_scores)
            # Logits mask
            logit_mask = np.zeros(self.total_classes)
            logit_mask[train_labels] = 1.0

            # Loop over the entire training dataset to compute the parameter importance
            batch_size = 10
            num_samples = train_x.shape[0]
            for iters in range(num_samples// batch_size):
                offset = iters * batch_size
                sess.run(self.accumulate_hebbian_scores, feed_dict={self.x: train_x[offset:offset+batch_size], self.keep_prob: 1.0, 
                    self.output_mask: logit_mask})

            # Average the hebbian scores across the training examples
            sess.run(self.average_hebbian_scores, feed_dict={self.train_samples: num_samples})
            
        # Store current weights
        self.init_updates(sess)

    def restore(self, sess):
        """
        Restore the weights from the star variables
        Args:
            sess        TF session

        Returns:
        """
        sess.run(self.restore_weights)
