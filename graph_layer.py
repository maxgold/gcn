## contains the model structure for the 
## graph convolution network
## Also has functions to score a network

from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
from tsp_utils import gen_cycle_data_missing, json_dic_to_np3, json_dic_to_np2
from tsp_utils import np_dic_to_json2, np_dic_to_json3
import os
import sys
import collections
import IPython as ipy
import pickle
#from gen_tsp import *



def graph_layer(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    out_layer = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])
    out_layer = tf.nn.relu(out_layer)

    return out_layer

def graph_layer2(x, hidden1, hidden2):
    ## NOTE: THIS FLATTENS x...not really what I had in mind
    layer_1   = tf.layers.dense(x, hidden1, use_bias=True)
    out_layer = tf.layers.dense(layer_1, hidden2, use_bias=True)

    return out_layer


class graph_conv_net_r(object):
    def __init__(self, layers, learning_rate):
        self.layers = layers
        self.learning_rate = learning_rate

    def create_layers(self, x, y, adj):
        pred = x
        for hidden in self.layers:
            ## This flattens pred then does something...not really what I had in mind
            pred  = tf.layers.dense(pred, hidden)
            pred  = tf.einsum('ijk,ilj->ilk', pred, adj) # i verified in numpy that this works


        self.cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, pred))))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

class graph_conv_net(object):
    def __init__(self, params):
        self.weights = params['weights']
        self.biases  = params['biases']
        self.relu    = params['relu']
        self.learning_rate = params['lr']
        self.activation = params['activation']
        self.params = params

    def create_layers(self, x, y, adj):
        pred = x
        L    = len(self.weights.keys())
        i    = 0
        self.cost = 0
        for w, b in zip(self.weights.keys(), self.biases.keys()):
            pred  = tf.add(tf.matmul(pred, self.weights[w]), self.biases[b])
            self.cost += .001*tf.nn.l2_loss(self.weights[w])
            self.cost += .001*tf.nn.l2_loss(self.biases[b])
            if i < L - 1:
                pred  = tf.nn.relu(pred)
            i += 1
            #pred  = tf.einsum('ijk,ilj->ilk', pred, adj) # i verified in numpy that this works

        self.pred = pred
        self.y    = y
        self.softmax = tf.nn.softmax(pred, dim=1)
        self.cost1 = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y, dim=1)
        self.cost += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y, dim=1))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)


### IDEA: Try reshaping the last layer to be standard 1d fully connected

class graph_conv_net_label(object):
    def __init__(self, params):
        self.weights       = params['weights']
        self.biases        = params['biases']
        self.weights_keys  = list(np.sort(list(self.weights.keys())))
        self.bias_keys     = list(np.sort(list(self.biases.keys())))
        self.relu          = params['relu']        
        self.global_step   = params['global_step']
        self.learning_rate = params['lr']
        self.params        = params
        self.weight_reg    = params['weight_reg']
        self.folder_path   = params['folder_path']
        self.model_name    = params['model_name']
        self.save_path     = self.folder_path + self.model_name + '/'
        self.saver         = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=.1)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)


    def create_layers(self, x, y, P1_tf, P2_tf, Ann_tf, F_tf):
        i    = 0
        ind = 0
        pred = {}
        pred[ind] = x
        L    = len(self.weights.keys())
        self.cost = 0
        for w, b in zip(self.weights_keys[:-1], self.bias_keys[:-1]):
            # could make a 2 layer net by adding another S
            P1   = P1_tf[i]
            P2   = P2_tf[i]
            A_nn = Ann_tf[i]
            F    = F_tf[i]
            I    = tf.concat((tf.matmul(P1, pred[ind]), tf.matmul(P2, pred[ind]), F), 1)
            skip = tf.concat((tf.matmul(P1, x), tf.matmul(P2, x)), 1)
            if self.params['skip']:
                I = tf.concat((I, skip), 1)
            S    = tf.add(tf.matmul(I, self.weights[w]), self.biases[b])
            S    = tf.nn.relu(S)
            ind += 1
            pred[ind] = tf.matmul(A_nn, S)
            pred[ind] = tf.nn.relu(pred[ind])
            self.cost += self.weight_reg*tf.nn.l2_loss(self.weights[w])
            self.cost += self.weight_reg*tf.nn.l2_loss(self.biases[b])
            i += 1
        w = self.weights_keys[-1]
        b = self.bias_keys[-1]
        pred[ind+1] = tf.add(tf.matmul(pred[ind], self.weights[w]), self.biases[b])
        ind += 1
        self.cost += self.weight_reg*tf.nn.l2_loss(self.weights[w])
        self.cost += self.weight_reg*tf.nn.l2_loss(self.biases[b])


        self.pred = pred[ind]
        self.y    = y
        self.softmax = tf.nn.softmax(self.pred, dim=0)
        self.cost1 = tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=y, dim=0)
        self.cost += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=y, dim=0))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost, global_step=self.global_step)
        #self.accuracy = tf.metrics.accuracy(labels=y, predictions=pred)

    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.cost)
            #tf.summary.scalar("accuracy", self.accuracy)
            self.summary_op = tf.summary.merge_all()

class graph_conv_net_reg(object):
    def __init__(self, params):
        self.weights       = params['weights']
        self.biases        = params['biases']
        self.weights_keys  = list(np.sort(list(self.weights.keys())))
        self.bias_keys     = list(np.sort(list(self.biases.keys())))
        self.relu          = params['relu']        
        self.global_step   = params['global_step']
        self.learning_rate = params['lr']
        self.activation    = params['activation']
        self.params        = params
        self.weight_reg    = params['weight_reg']
        self.folder_path   = params['folder_path']
        self.model_name    = params['model_name']
        self.save_path     = self.folder_path + self.model_name + '/'
        self.saver         = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=.1)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)


    def create_layers(self, x, y1, y2, P1_tf, P2_tf, Ann_tf, F_tf, index_tf):
        pred = x
        L    = len(self.weights.keys())
        i    = 0
        self.cost = 0
        for w, b in zip(self.weights_keys[:-1], self.bias_keys[:-1]):
            # could make a 2 layer net by adding another S
            P1   = P1_tf[i]
            P2   = P2_tf[i]
            A_nn = Ann_tf[i]
            F    = F_tf[i]
            I    = tf.concat((tf.matmul(P1, pred), tf.matmul(P2, pred), F), 1)
            skip = tf.concat((tf.matmul(P1, x), tf.matmul(P2, x)), 1)
            if self.params['skip']:
                I = tf.concat((I, skip), 1)
            S    = tf.add(tf.matmul(I, self.weights[w]), self.biases[b])
            S    = tf.nn.relu(S)
            pred = tf.matmul(A_nn, S)
            pred = tf.nn.relu(pred)
            self.cost += self.weight_reg*tf.nn.l2_loss(self.weights[w])
            self.cost += self.weight_reg*tf.nn.l2_loss(self.biases[b])
            i += 1

        w = self.weights_keys[-1]
        b = self.bias_keys[-1]
        pred = tf.add(tf.matmul(pred, self.weights[w]), self.biases[b])

        self.cost += self.weight_reg*tf.nn.l2_loss(self.weights[w])
        self.cost += self.weight_reg*tf.nn.l2_loss(self.biases[b])

        val1 = y1
        val2 = y2
        self.pred = pred
        self.y    = y2
        pred_val = pred[index_tf]
        self.pred_val = pred_val
        self.cost1 = tf.losses.absolute_difference(val1, pred_val)
        self.cost2 = tf.losses.mean_squared_error(val1, pred_val)
        if self.params['loss'] == 'l2':
            self.cost += tf.losses.mean_squared_error(val1, pred_val)
            self.cost += .01*tf.losses.mean_squared_error(val2, pred)
            self.cost -= .01*tf.losses.mean_squared_error(val2, pred_val)
        else:
            self.cost += tf.losses.absolute_difference(val1, pred_val)  
            self.cost += tf.losses.absolute_difference(val2, pred)
            self.cost -= tf.losses.absolute_difference(val2, pred_val)      
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost, global_step=self.global_step)
        #self.accuracy = tf.metrics.accuracy(labels=y, predictions=pred)
# Parameters

class graph_conv_net_ranking(object):
    def __init__(self, params):
        self.weights       = params['weights']
        self.biases        = params['biases']
        self.weights_keys  = list(np.sort(list(self.weights.keys())))
        self.bias_keys     = list(np.sort(list(self.biases.keys())))
        self.relu          = params['relu']        
        self.global_step   = params['global_step']
        self.learning_rate = params['lr']
        self.activation    = params['activation']
        self.params        = params
        self.weight_reg    = params['weight_reg']
        self.folder_path   = params['folder_path']
        self.model_name    = params['model_name']
        self.save_path     = self.folder_path + self.model_name + '/'
        self.saver         = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=.1)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)


    def create_layers(self, x, y, P1_tf, P2_tf, Ann_tf, F_tf, visited):
        pred = x
        L    = len(self.weights.keys())
        i    = 0
        self.cost = 0
        for w, b in zip(self.weights_keys[:-1], self.bias_keys[:-1]):
            # could make a 2 layer net by adding another S
            P1   = P1_tf[i]
            P2   = P2_tf[i]
            A_nn = Ann_tf[i]
            F    = F_tf[i]
            I    = tf.concat((tf.matmul(P1, pred), tf.matmul(P2, pred), F), 1)
            skip = tf.concat((tf.matmul(P1, x), tf.matmul(P2, x)), 1)
            if self.params['skip']:
                I = tf.concat((I, skip), 1)
            S    = tf.add(tf.matmul(I, self.weights[w]), self.biases[b])
            S    = tf.nn.relu(S)
            pred = tf.matmul(A_nn, S)
            pred = tf.nn.relu(pred)
            self.cost += self.weight_reg*tf.nn.l2_loss(self.weights[w])
            self.cost += self.weight_reg*tf.nn.l2_loss(self.biases[b])
            i += 1

        w = self.weights_keys[-1]
        b = self.bias_keys[-1]
        pred = tf.add(tf.matmul(pred, self.weights[w]), self.biases[b])



        self.cost += self.weight_reg*tf.nn.l2_loss(self.weights[w])
        self.cost += self.weight_reg*tf.nn.l2_loss(self.biases[b])

        self.pred  = visited*pred
        self.y     = 3 * visited*y
        self.cost1 = tf.losses.absolute_difference(self.y, pred)
        self.cost2 = tf.losses.mean_squared_error(self.y, pred)
        if self.params['loss'] == 'l2':
            self.cost += tf.losses.mean_squared_error(self.y, pred)
        else:
            self.cost += tf.losses.absolute_difference(self.y, pred)  
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost, global_step=self.global_step)
        #self.accuracy = tf.metrics.accuracy(labels=y, predictions=pred)
# Parameters


def test_learn_sum():
    learning_rate   = 0.001
    training_epochs = 100
    batch_size      = 100
    display_step    = 1
    num_nodes       = 15

    n_input = 6
    n_out   = 1
    # tf Graph input
    x = tf.placeholder("float", [None, num_nodes, n_input])
    y = tf.placeholder("float", [None, num_nodes, n_out])
    adj_mat = tf.placeholder("float", [None, num_nodes, num_nodes])

    X = np.random.rand(10000, num_nodes, 6)
    Y = X.sum(axis=2)[:, :,None]

    X_test = np.random.rand(100, num_nodes, 6)
    Y_test = X_test.sum(axis=2)[:, :,None]

    layers = [6, 1]

    with tf.Session() as sess:
        model = graph_conv_net_r(layers, learning_rate)
        model.create_layers(x, y, adj_mat)

        init = tf.global_variables_initializer()
        sess.run(init)
        i = 0
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(len(X)/batch_size)
            adj = np.tile(np.eye(num_nodes), (batch_size, 1,1))

            # Loop over all batches
            for i in range(total_batch):
                batch_x = X[i*batch_size:(i+1)*batch_size]
                batch_y = Y[i*batch_size:(i+1)*batch_size]

                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([model.optimizer, model.cost], feed_dict={x: batch_x,
                                                              y: batch_y, adj_mat:adj})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
        print("Optimization Finished!")
        c = sess.run(model.cost, feed_dict={x: X_test, y: Y_test, adj_mat:adj})
        print('Final cost', c)

    return(model)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(2))
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    t = np.tile(np.eye(adj.shape[1]), (adj.shape[0],1,1))*d_inv_sqrt[:,:,None]
    temp = np.zeros(adj.shape)
    for z in range(adj.shape[0]):
        temp1 = adj[z, :, :]
        temp2 = t[z,:,:]
        temp[z,:,:] = (temp2.dot(temp1)).dot(temp2)

    return temp

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    N = adj.shape[2]
    eye = np.repeat(np.eye(N)[None, :,:],adj.shape[0],axis=0)
    adj_normalized = normalize_adj(adj + eye)
    return adj_normalized

def construct_weights_biases(size):
    weights = {}
    biases  = {}
    for i in range(len(size)-2):
        name_w = 'h' + str(i)
        name_b = 'b' + str(i)
        weights[name_w] = tf.Variable(tf.random_normal([2*size[i] + 2, size[i+1]]))
        biases[name_b]  = tf.Variable(tf.random_normal([size[i+1]]))
    i += 1
    name_w = 'h' + str(i)
    name_b = 'b' + str(i)
    weights[name_w] = tf.Variable(tf.random_normal([size[i], size[i+1]]))
    biases[name_b]  = tf.Variable(tf.random_normal([size[i+1]]))


    return(weights, biases)

def construct_weights_biases_skip(size):
    weights = {}
    biases  = {}
    n_input = size[0]
    for i in range(len(size)-2):
        name_w = 'h' + str(i)
        name_b = 'b' + str(i)
        weights[name_w] = tf.Variable(tf.random_normal([2*size[i] + 2*n_input + 2, size[i+1]]))
        biases[name_b]  = tf.Variable(tf.random_normal([size[i+1]]))
    i += 1
    name_w = 'h' + str(i)
    name_b = 'b' + str(i)
    weights[name_w] = tf.Variable(tf.random_normal([size[i], size[i+1]]))
    biases[name_b]  = tf.Variable(tf.random_normal([size[i+1]]))


    return(weights, biases)

def construct_weights_biases_reg(size, num_nodes):
    weights = {}
    biases  = {}
    for i in range(len(size)-2):
        name_w = 'h' + str(i)
        name_b = 'b' + str(i)
        weights[name_w] = tf.Variable(tf.random_normal([2*size[i] + 2, size[i+1]]))
        biases[name_b]  = tf.Variable(tf.random_normal([size[i+1]]))
    i += 1
    name_w = 'h' + str(i)
    name_b = 'b' + str(i)
    weights[name_w] = tf.Variable(tf.random_normal([1, num_nodes]))
    biases[name_b]  = tf.Variable(tf.random_normal([1]))


    return(weights, biases)



def vars_from_data_dict(data_dict):
  features = json_dic_to_np2(data_dict['features'])
  routes = json_dic_to_np2(data_dict['routes'])
  P1 = json_dic_to_np3(data_dict['P1'])
  P2 = json_dic_to_np3(data_dict['P2'])
  A_nn = json_dic_to_np3(data_dict['A_nn'])
  F = json_dic_to_np3(data_dict['F'])
  ws = json_dic_to_np2(data_dict['ws'])
  node_list = data_dict['node_list']
  field_size = data_dict['field_size']
  optimal = json_dic_to_np2(data_dict['optimal'])

  return([features, routes, P1, P2, A_nn, F, ws, field_size, node_list, optimal])

def save_model(sess, model,gs):
    if gs % 10000 == 0 & gs > 50000:
        model.saver.save(sess, model.save_path, global_step = gs, write_meta_graph = False)
    elif gs % 10000 == 0:
        model.saver.save(sess, model.save_path, global_step = gs, write_meta_graph = True)

def params_to_name(mode, node_list, field_size, sizes, num_train_ex, num_test_ex):
  name = 'graph-' + mode + '-'
  for node in node_list:
    name += str(node)
  name += '-'
  for fs in field_size:
    name += str(fs)
  name += '-'
  for size in sizes:
    name += str(size)
  name += '-'
  name += str(num_train_ex)
  name += '-'
  name += str(num_test_ex)
  name += '.p'
  return(name)



def train_model_class(sess, model, data, placeholders, field_size, train_inds, test_inds, node_list,training_epochs = 20, batch_size=1,display_step=1, early_stop = 0):
    #model._create_summaries()
    features, routes, P1, P2, A_nn, F, ws, optimal, dist_left, rankings = data
    x, y, P1_tf, P2_tf, Ann_tf, F_tf, global_step, visited = placeholders
    keys       = list(features.keys())
    tot_ex     = len(features.keys())
    num_layers = len(field_size)
    epoch_res  = []

    log_loss_recent = collections.deque(maxlen=5)
    #writer = tf.summary.FileWriter(model.save_path, sess.graph, flush_secs=60,filename_suffix='.summary')

    
    for epoch in range(training_epochs):
        start = time.time()
        avg_cost = 0.
        total    = 0.

        # Loop over all batches
        np.random.shuffle(train_inds)
        for ind in train_inds:
            num_nodes = features[keys[ind]][0].shape[0]
            start1 = time.time()
            for in_ind in features[keys[ind]].keys():
                batch_x  = features[keys[ind]][in_ind]
                batch_y  = routes[keys[ind]][in_ind]

                P1_feed  = P1[keys[ind]][in_ind]
                P2_feed  = P2[keys[ind]][in_ind]
                Ann_feed = A_nn[keys[ind]][in_ind]
                F_feed   = F[keys[ind]][in_ind]

                for i in range(batch_x.shape[2]):
                    x_feed = batch_x[:,:,i]
                    x_feed = x_feed / x_feed.sum(0)
                    #x_feed = x_feed/num_nodes
                    y_feed = np.zeros([num_nodes, 1])
                    y_feed[batch_y[i]] = 1


                    feed_dict = {}
                    for layer in range(num_layers):
                        feed_dict[P1_tf[layer]] = P1_feed[layer]
                        feed_dict[P2_tf[layer]] = P2_feed[layer]
                        feed_dict[Ann_tf[layer]] = Ann_feed[layer]
                        feed_dict[F_tf[layer]]   = F_feed[layer]

                    feed_dict[x] = x_feed
                    feed_dict[y] = y_feed

                    _, pred, y_t, softmax, c1, c = sess.run([model.optimizer, model.pred, model.y,
                                                    model.softmax, model.cost1, model.cost], 
                                                        feed_dict=feed_dict)
                    
                    # print(in_ind, i)
                    # print(c1)
                    # if c1[0] > .1:
                    #    ipy.embed()
                    #avg_cost = .01*c1[0] + .99*avg_cost
                    avg_cost += c

                    total    += 1
                    gs = sess.run(global_step)

                    # Compute average loss
            end1 = time.time()
            #print('inner took ' + str(end1-start1) + ' seconds')
                
        # Display logs per epoch step
        end = time.time()
        if epoch % display_step == 0:
            print("Cost:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost/total))
            print('Epoch took ' + str(end-start) + ' seconds')
            print('Learning rate is ' + str(sess.run(model.learning_rate)))
            try:
                res = score_model_class(sess, model, data, placeholders, field_size, node_list, test_inds)
                for node in res.keys():
                    print(node)
                    print(res[node])
            except:
                None
        log_loss_recent.append(avg_cost/total)
        if np.all(np.abs(np.diff(np.array(log_loss_recent))) < .01) & epoch > 10:
            break
        if avg_cost/total < early_stop:
            break

def train_model_ranking(sess, model, data, placeholders, field_size, train_inds, test_inds, node_list,training_epochs = 20, batch_size=1,display_step=1):
    #model._create_summaries()
    features, routes, P1, P2, A_nn, F, ws, optimal, dist_left, rankings = data
    x, y, P1_tf, P2_tf, Ann_tf, F_tf, global_step, visited = placeholders
    keys       = list(features.keys())
    tot_ex     = len(features.keys())
    num_layers = len(field_size)
    epoch_res  = []

    log_loss_recent = collections.deque(maxlen=5)
    #writer = tf.summary.FileWriter(model.save_path, sess.graph, flush_secs=60,filename_suffix='.summary')

    
    for epoch in range(training_epochs):
        start = time.time()
        avg_cost = 0.
        total    = 0.

        # Loop over all batches
        np.random.shuffle(train_inds)
        for ind in train_inds:
            num_nodes = features[keys[ind]][0].shape[0]
            start1 = time.time()
            for in_ind in range(num_nodes):
                batch_x  = features[keys[ind]][in_ind]
                batch_y  = rankings[keys[ind]][in_ind]

                P1_feed  = P1[keys[ind]][in_ind]
                P2_feed  = P2[keys[ind]][in_ind]
                Ann_feed = A_nn[keys[ind]][in_ind]
                F_feed   = F[keys[ind]][in_ind]

                for i in range(batch_x.shape[2]):
                    x_feed = batch_x[:,:,i]
                    x_feed = x_feed / x_feed.sum(0)
                    y_feed = batch_y[i,:][:,None]
                    visited_feed = (x_feed[:,2] > 0).astype(int)[:,None]

                    feed_dict = {}
                    for layer in range(num_layers):
                        feed_dict[P1_tf[layer]] = P1_feed[layer]
                        feed_dict[P2_tf[layer]] = P2_feed[layer]
                        feed_dict[Ann_tf[layer]] = Ann_feed[layer]
                        feed_dict[F_tf[layer]]   = F_feed[layer]

                    feed_dict[x] = x_feed
                    feed_dict[y] = y_feed
                    feed_dict[visited] = 1 - visited_feed

                    _, pred, y_t, c1, c = sess.run([model.optimizer, model.pred, model.y,
                                                    model.cost1, model.cost], 
                                                        feed_dict=feed_dict)
                    
                    # print(in_ind, i)
                    # print(c1)
                    # if c1[0] > .1:
                    #    ipy.embed()
                    #avg_cost = .01*c1[0] + .99*avg_cost
                    avg_cost += c1

                    total    += 1
                    gs = sess.run(global_step)
                    save_model(sess, model, gs)

                    # Compute average loss
            end1 = time.time()
            #print('inner took ' + str(end1-start1) + ' seconds')
                
        # Display logs per epoch step
        end = time.time()
        if epoch % display_step == 0:
            print("Cost:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost/total))
            print('Epoch took ' + str(end-start) + ' seconds')
            print('Learning rate is ' + str(sess.run(model.learning_rate)))
            try:
                res = score_model_class(sess, model, data, placeholders, field_size, node_list, test_inds)
                for node in res.keys():
                    print(node)
                    print(res[node])
            except:
                None

def train_model_reg(sess, model, data, placeholders, field_size, train_inds,  training_epochs = 20, batch_size=1,display_step=1):
    features, routes, P1, P2, A_nn, F, ws, optimal, dist_left = data
    x, y1, y2, P1_tf, P2_tf, Ann_tf, F_tf, global_step, index_tf = placeholders
    keys       = list(features.keys())
    tot_ex     = len(features.keys())
    num_layers = len(field_size)
    epoch_res  = []

    #writer = tf.summary.FileWriter(model.save_path, sess.graph, flush_secs=60,filename_suffix='.summary')


    for epoch in range(training_epochs):
        start = time.time()
        avg_cost1 = 0.
        avg_cost2 = 0.
        avg_pred  = 0.
        avg_rel   = 0.
        total    = 0.

        # Loop over all batches
        np.random.shuffle(train_inds)
        for ind in train_inds:
            num_nodes = features[keys[ind]][0].shape[0]
            start1 = time.time()
            for in_ind in range(num_nodes):
                batch_x  = features[keys[ind]][in_ind]
                batch_y  = np.array(dist_left[keys[ind]][in_ind])
                P1_feed  = P1[keys[ind]][in_ind]
                P2_feed  = P2[keys[ind]][in_ind]
                Ann_feed = A_nn[keys[ind]][in_ind]
                F_feed   = F[keys[ind]][in_ind]

                for i in range(batch_x.shape[2]):
                    x_feed = batch_x[:,:,i]
                    #x_feed = x_feed / x_feed.sum(1)[:, None]
                    x_feed = x_feed / x_feed.sum(0)
                    #x_feed = x_feed/num_nodes

                    y_feed1 = batch_y[i, 1]
                    y_feed2 = batch_y[i, 2]
                    index_feed = batch_y[i, 0]

                    feed_dict = {}
                    for layer in range(num_layers):
                        feed_dict[P1_tf[layer]] = P1_feed[layer]
                        feed_dict[P2_tf[layer]] = P2_feed[layer]
                        feed_dict[Ann_tf[layer]] = Ann_feed[layer]
                        feed_dict[F_tf[layer]]   = F_feed[layer]

                    feed_dict[x] = x_feed
                    feed_dict[y1] = y_feed1
                    feed_dict[y2] = y_feed2
                    feed_dict[index_tf] = index_feed

                    _, pred, y_t, c1, c2 = sess.run([model.optimizer, model.pred, 
                                                    model.y, model.cost1, model.cost2], 
                                                    feed_dict=feed_dict)

                    avg_cost1 += c1
                    avg_cost2 += c2
                    avg_pred  += y_feed1
                    if y_feed1 > 1e-2:
                        avg_rel   += c1/(y_feed1)
                    else:
                        avg_rel += avg_rel/total
                    total    += 1
                    gs = sess.run(global_step)

                    #save_model(sess, model, gs)

                    # Compute average loss
            end1 = time.time()
            #print('inner took ' + str(end1-start1) + ' seconds')
                
        # Display logs per epoch step
        end = time.time()
        if epoch % display_step == 0:
            print("Cost L1:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost1/total))
            print("Cost L2:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost2/total))
            print("Pred avg:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_pred/total))
            print("Rel avg:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_rel/total))

            print('Epoch took ' + str(end-start) + ' seconds')
            print('Learning rate is ' + str(sess.run(model.learning_rate)))


def score_model_class(sess, model, data, placeholders, field_size, node_list, test_inds):
    features, routes, P1, P2, A_nn, F, ws, optimal, dist_left, rankings = data
    x, y, P1_tf, P2_tf, Ann_tf, F_tf, global_step, visited = placeholders
    keys       = list(features.keys())
    tot_ex     = len(features.keys())
    num_layers = len(field_size)

    test_inds              = np.sort(test_inds)
    solved_strict          = 0
    unsolved_strict        = 0
    solved_total           = 0
    unsolved_total         = 0
    solved_min             = 0
    unsolved_min           = 0
    greedy_solved_strict   = 0
    greedy_unsolved_strict = 0
    greedy_solved_min      = 0
    greedy_unsolved_min    = 0
    unsolved_key           = []
    solved_key             = []
    solved_cost            = {}
    optimal_cost           = {}
    greedy_cost            = {}
    for node in node_list:
        solved_cost[node]  = {}
        optimal_cost[node] = {}
        greedy_cost[node]   = {}

    for ind in test_inds:
        #print(ind)
        key = keys[ind]
        num_nodes = key[1]
        solved_cost[num_nodes][ind]  = []
        optimal_cost[num_nodes][ind] = []
        greedy_cost[num_nodes][ind]  = []

        
        for in_ind in range(num_nodes):
            batch_x  = features[keys[ind]][in_ind]
            batch_y  = routes[keys[ind]][in_ind]
            P1_feed  = P1[keys[ind]][in_ind]
            P2_feed  = P2[keys[ind]][in_ind]
            Ann_feed = A_nn[keys[ind]][in_ind]
            F_feed   = F[keys[ind]][in_ind]
            opt_cost = optimal[keys[ind]][in_ind]
            W        = ws[keys[ind]][in_ind]
            greedy   = greedy_tsp(num_nodes, W, in_ind)
            if greedy == -1:
                greedy_unsolved_strict += 1
            else:
                greedy_solved_strict += 1

            start_node = in_ind
            goal_node  = in_ind

            visited = np.zeros(num_nodes)
            visited[start_node] = 1

            x_feed = batch_x[:,:,0]
            x_feed = x_feed / x_feed.sum(0)
            #x_feed = x_feed/num_nodes
            goal0  = batch_x[:, 3:, 0]
            y_feed = np.zeros([num_nodes, 1])
            feed_dict = {}
            for layer in range(num_layers):
                feed_dict[P1_tf[layer]] = P1_feed[layer]
                feed_dict[P2_tf[layer]] = P2_feed[layer]
                feed_dict[Ann_tf[layer]] = Ann_feed[layer]
                feed_dict[F_tf[layer]]   = F_feed[layer]

            cost = 0
            steps = 0
            cur_node = start_node
            solved_bool = False
            while steps < 30:
                feed_dict[x] = x_feed
                feed_dict[y] = y_feed
                visited_loc = list(np.where(visited==1)[0])
                visited_loc = [a for a in visited_loc if a != start_node]
                row = W[cur_node]

                softmax = sess.run([model.softmax], feed_dict=feed_dict)
                softmax = softmax[0][:,0]
                #print(np.max(softmax))
                steps += 1
                cand = [n for n in np.argsort(-softmax) if (n not in visited_loc) and row[n]!=0]
                if len(cand) > 0:
                    next_node = cand[0]
                else:
                    break
                if W[cur_node, next_node] == 0:
                    print(ind, in_ind)
                    print('ahhh')
                #print(next_node)
                state0 = np.zeros([num_nodes, 3])
                state0[next_node, 0] = 1
                state0[goal_node, 1] = 1
                visited[next_node] = 1
                state0[:, 2] = visited
                cost += W[cur_node, next_node]
                cur_node = next_node

                if (next_node == start_node) & (np.sum(visited)==num_nodes):
                    if steps <= num_nodes:
                        solved_strict += 1
                    if steps > num_nodes:
                        unsolved_strict += 1
                    solved_total  += 1
                    solved_key.append(keys[ind])
                    solved_cost[num_nodes][ind].append(cost)
                    optimal_cost[num_nodes][ind].append(opt_cost)
                    greedy_cost[num_nodes][ind].append(greedy)
                    solved_bool = True
                    break
                x_feed = np.c_[state0, goal0]
                x_feed = x_feed / x_feed.sum(0)
                #x_feed = x_feed/num_nodes

            if not solved_bool:
                #print('Unsolved')
                #print(ind, in_ind)
                unsolved_strict += 1
                unsolved_total += 1
                unsolved_key.append(keys[ind])

    subopt = {}
    for num_nodes in node_list:
        greedy_cost1 = []
        solved_cost1 = []
        optimal_cost1 = []
        sc1 = []
        oc1 = []
        gc1 = []
        #import IPython as ipy
        #ipy.embed()
        for ind in solved_cost[num_nodes].keys():
            greedy_cost1.extend(greedy_cost[num_nodes][ind])
            solved_cost1.extend(solved_cost[num_nodes][ind])
            optimal_cost1.extend(optimal_cost[num_nodes][ind])
            if len(solved_cost[num_nodes][ind]) > 0:
                solved_min += 1
                greedy_success = [x for x in greedy_cost[num_nodes][ind] if x > 0]
                if len(greedy_success) > 0:
                    greedy_solved_min += 1
                else:
                    greedy_unsolved_min += 1
                if len(greedy_success) > 0:
                    greedy_val = min(greedy_success)
                else:
                    greedy_val = .5*num_nodes
                gc1.append(greedy_val)
                sc1.append(min(solved_cost[num_nodes][ind]))
                oc1.append(min(optimal_cost[num_nodes][ind]))
            else:
                unsolved_min += 1

        #print(np.mean(solved_cost1 - optimal_cost1)/optimal_cost1)
        greedy_cost1 = np.array(greedy_cost1)
        optimal_cost1 = np.array(optimal_cost1)
        solved_cost1 = np.array(solved_cost1)
        gc1 = np.array(gc1)
        oc1 = np.array(oc1)
        sc1 = np.array(sc1)
        subopt[num_nodes] = []
        greedy_mask = (greedy_cost1 > 0)


        subopt[num_nodes].append(np.mean((greedy_cost1[greedy_mask]-optimal_cost1[greedy_mask])/optimal_cost1[greedy_mask]))
        subopt[num_nodes].append(np.mean((np.array(gc1)-np.array(oc1))/np.array(oc1)))

        subopt[num_nodes].append(np.mean((solved_cost1-optimal_cost1)/optimal_cost1))
        subopt[num_nodes].append(np.mean((sc1-oc1)/oc1))
        #print(np.mean((sc1 - oc1)/oc1))
    subopt['unsolved_strict'] = unsolved_strict
    subopt['solved_strict'] = solved_strict
    subopt['unsolved_total'] = unsolved_total
    subopt['solved_total'] = solved_total
    subopt['unsolved_min'] = unsolved_min
    subopt['solved_min']   = solved_min
    subopt['greedy_unsolved_strict'] = greedy_unsolved_strict
    subopt['greedy_solved_strict'] = greedy_solved_strict
    subopt['greedy_unsolved_min'] = greedy_unsolved_min
    subopt['greedy_solved_min']   = greedy_solved_min

    return(subopt)

def score_model_reg(sess, model, data, placeholders, field_size, test_inds, training_epochs = 20, batch_size=1,display_step=1):
    features, routes, P1, P2, A_nn, F, ws, optimal, dist_left = data
    x, y1, y2, P1_tf, P2_tf, Ann_tf, F_tf, global_step, index_tf = placeholders
    keys       = list(features.keys())
    tot_ex     = len(features.keys())
    num_layers = len(field_size)
    epoch_res  = []

    avg_cost1 = 0.
    avg_cost2 = 0.
    avg_pred  = 0.
    avg_rel   = 0.
    total    = 0.

    for ind in test_inds:
        num_nodes = features[keys[ind]][0].shape[0]
        start1 = time.time()
        for in_ind in range(num_nodes):
            batch_x  = features[keys[ind]][in_ind]
            batch_y  = np.array(dist_left[keys[ind]][in_ind])
            P1_feed  = P1[keys[ind]][in_ind]
            P2_feed  = P2[keys[ind]][in_ind]
            Ann_feed = A_nn[keys[ind]][in_ind]
            F_feed   = F[keys[ind]][in_ind]

            for i in range(batch_x.shape[2]):
                x_feed = batch_x[:,:,i]
                #x_feed = x_feed / x_feed.sum(1)[:, None]
                x_feed = x_feed / x_feed.sum(0)
                #x_feed = x_feed/num_nodes

                y_feed1 = batch_y[i, 1]
                y_feed2 = batch_y[i, 2]
                index_feed = batch_y[i, 0]

                feed_dict = {}
                for layer in range(num_layers):
                    feed_dict[P1_tf[layer]] = P1_feed[layer]
                    feed_dict[P2_tf[layer]] = P2_feed[layer]
                    feed_dict[Ann_tf[layer]] = Ann_feed[layer]
                    feed_dict[F_tf[layer]]   = F_feed[layer]

                feed_dict[x] = x_feed
                feed_dict[y1] = y_feed1
                feed_dict[y2] = y_feed2

                feed_dict[index_tf] = index_feed

                pred, y_t, c1, c2 = sess.run([model.pred, model.y, model.cost1, 
                                            model.cost2], feed_dict=feed_dict)
                #import IPython as ipy
                #ipy.embed()
                avg_cost1 += c1
                avg_cost2 += c2
                avg_pred += y_feed1
                if y_feed1 > 1e-2:
                    avg_rel   += c1/(y_feed1)
                else:
                    avg_rel += avg_rel/total

                total    += 1


            
    print("cost l1=", \
        "{:.9f}".format(avg_cost1/total))
    print("cost l2=", \
        "{:.9f}".format(avg_cost2/total))
    print("Pred avg=", \
        "{:.9f}".format(avg_pred/total))
    print("Rel avg=", \
        "{:.9f}".format(avg_rel/total))



def data_dict_to_vars(data_dict):
    features = data_dict[b'features']
    routes = data_dict[b'routes']
    P1 = data_dict[b'P1']
    P2 = data_dict[b'P2']
    A_nn = data_dict[b'A_nn']
    F = data_dict[b'F']
    ws = data_dict[b'ws']
    optimal = data_dict[b'optimal']
    dist_left = data_dict[b'dist_left']

    return(features, routes, P1, P2, A_nn, F, ws, optimal, dist_left)



def experiment(exp_params, data, data_test):
    node_list       = exp_params['node_list']
    field_size      = exp_params['field_size']
    sizes           = exp_params['sizes']
    skip            = exp_params['skip']
    lr_decay        = exp_params['lr_decay']
    weight_reg      = exp_params['weight_reg']
    num_train_ex    = exp_params['num_train_ex']
    num_test_ex     = exp_params['num_test_ex']
    added_edges     = exp_params['added_edges']
    training_epochs = exp_params['training_epochs']
    batch_size      = exp_params['batch_size']
    display_step    = exp_params['display_step']
    starter_learning_rate = exp_params['starter_learning_rate']


    num_layers = len(field_size)
    assert(len(sizes) == (num_layers+2))
    
    features, routes, P1, P2, A_nn, F, ws, optimal, dist_left, rankings = data
    tot_ex = len(features.keys())
    train_size = int(.8*tot_ex)

    inds       = np.random.permutation(tot_ex)
    train_inds = inds[:train_size]
    test_inds  = inds[train_size:]

    n_input = 6
    n_out  = 1

    if skip:
        weights, biases = construct_weights_biases_skip(sizes)
    else:
        weights, biases = construct_weights_biases(sizes)


    decay_step = int(train_size*np.mean(np.array(node_list))**2/3)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                decay_step, lr_decay, staircase=True)

    params = {}
    params['global_step'] = global_step
    params['lr'] = learning_rate
    params['relu'] = {
        'h1': True,
        'out': False
    }
    params['weights'] = weights
    params['biases'] = biases
    params['weight_reg'] = .001
    params['folder_path'] = '/Users/maxgold/rll/planning_networks/models/'
    params['model_name'] = 'test_model'
    params['skip'] = skip

    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, 1])
    visited = tf.placeholder("float", [None, 1])

    P1_tf   = {}
    P2_tf   = {}
    Ann_tf  = {}
    F_tf    = {}
    for i in range(num_layers):
        P1_tf[i]   = tf.placeholder('float', [None, None], name='P1')
        P2_tf[i]   = tf.placeholder('float', [None, None], name='P2')
        Ann_tf[i]  = tf.placeholder('float', [None, None], name='Ann')
        F_tf[i]    = tf.placeholder('float', [None, None], name='F')

    placeholders = [x, y, P1_tf, P2_tf, Ann_tf, F_tf, global_step, visited]

    sess = tf.Session()
    model = graph_conv_net_label(params)
    model.create_layers(x,y,P1_tf,P2_tf,Ann_tf, F_tf)
    init = tf.global_variables_initializer()
    sess.run(init)

    train_model_class(sess, model, data, placeholders, field_size, train_inds, 
                     test_inds, node_list, training_epochs, batch_size, display_step)

    test_inds = range(len(node_list)*(num_test_ex))
    res = score_model_class(sess, model, data_test, placeholders, field_size, node_list, test_inds)
    print(res)
    return(res, model)

def experiment_ec2(params, data_path):
    node_list       = params['node_list']
    field_size      = params['field_size']
    sizes           = params['sizes']
    skip            = params['skip']
    lr_decay        = params['lr_decay']
    weight_reg      = params['weight_reg']
    num_train_ex    = params['num_train_ex']
    num_test_ex     = params['num_test_ex']
    added_edges     = params['added_edges']
    training_epochs = params['training_epochs']
    batch_size      = params['batch_size']
    display_step    = params['display_step']
    starter_learning_rate = params['starter_learning_rate']
    num_layers = len(field_size)
    assert(len(sizes) == (num_layers+2))

    file_name = params_to_name('train', node_list, field_size, sizes, num_train_ex, num_test_ex)
    data_dict = pickle.load(open(data_path+file_name, 'rb'), encoding='bytes')
    data = data_dict_to_vars(data_dict)
    features, routes, P1, P2, A_nn, F, ws, optimal, dist_left = data
    tot_ex = len(features.keys())
    train_size = int(.8*tot_ex)

    inds       = np.random.permutation(tot_ex)
    train_inds = inds[:train_size]
    test_inds  = inds[train_size:]

    n_input = 6
    n_out  = 1

    if skip:
        weights, biases = construct_weights_biases_skip(sizes)
    else:
        weights, biases = construct_weights_biases(sizes)


    decay_step = int(train_size*np.mean(np.array(node_list))**2/3)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                decay_step, lr_decay, staircase=True)

    params = {}
    params['global_step'] = global_step
    params['lr'] = learning_rate
    params['relu'] = {
        'h1': True,
        'out': False
    }
    params['weights'] = weights
    params['biases'] = biases
    params['weight_reg'] = .001
    params['folder_path'] = '/Users/maxgold/rll/planning_networks/models/'
    params['model_name'] = 'test_model'
    params['skip'] = skip

    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, 1])
    P1_tf   = {}
    P2_tf   = {}
    Ann_tf  = {}
    F_tf    = {}
    for i in range(num_layers):
        P1_tf[i]   = tf.placeholder('float', [None, None], name='P1')
        P2_tf[i]   = tf.placeholder('float', [None, None], name='P2')
        Ann_tf[i]  = tf.placeholder('float', [None, None], name='Ann')
        F_tf[i]    = tf.placeholder('float', [None, None], name='F')

    placeholders = [x, y, P1_tf, P2_tf, Ann_tf, F_tf, global_step]

    sess = tf.Session()
    model = graph_conv_net_label(params)
    model.create_layers(x,y,P1_tf,P2_tf,Ann_tf, F_tf)
    init = tf.global_variables_initializer()
    sess.run(init)



    train_model_class(sess, model, data, placeholders, field_size, train_inds, 
                     test_inds, node_list, training_epochs, batch_size, display_step)

    file_name = params_to_name('test', node_list, field_size, sizes, num_train_ex, num_test_ex)
    data_dict_test = pickle.load(open(data_path+file_name, 'rb'), encoding='bytes')
    data_test = data_dict_to_vars(data_dict_test)
    test_inds = range(num_test_ex)
    res = score_model_class(sess, model, data_test, placeholders, field_size, node_list, test_inds)
    print(res)
    return(res)


def greedy_tsp(num_nodes, W, start_node):
    visited = [start_node]
    cur_node = start_node
    cost = 0
    steps = 0
    solved = False
    while steps < 30:
        row = W[visited[-1]]
        cand = [x for x in np.argsort(row) if (x not in visited) and row[x]!=0]
        if W[visited[-1], start_node] != 0:
            cand.append(start_node)
        if len(cand) > 0:
            next_node = cand[0]
        else:
            break
        visited.append(next_node)
        cost += W[cur_node, next_node]
        if (next_node == start_node) & (len(visited)==num_nodes+1):
            solved = True
            break
        cur_node = next_node
        steps += 1

    if solved:
        cost += W[visited[-1], start_node]
        return(cost)
    else:
        return(-1)









