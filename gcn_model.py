## contains the model structure for the 
## graph convolution network

import IPython as ipy
import tensorflow as tf
import numpy as np
from graph_layer import greedy_tsp
import copy
import time

class gcn_net_label(object):
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


    def create_layers(self, x, y, A_tf):
        ind = 0
        pred = {}
        pred[ind] = x
        L    = len(self.weights.keys())
        self.cost = 0
        for w, b in zip(self.weights_keys, self.bias_keys):
            # could make a 2 layer net by adding another S
            I    = tf.matmul(A_tf, pred[ind])
            skip = tf.matmul(A_tf, x)
            if self.params['skip']:
                I = tf.concat((I, skip), 1)
            S    = tf.add(tf.matmul(I, self.weights[w]), self.biases[b])
            S    = tf.nn.relu(S)
            ind += 1
            pred[ind] = S
            self.cost += self.weight_reg*tf.nn.l2_loss(self.weights[w])
            self.cost += self.weight_reg*tf.nn.l2_loss(self.biases[b])


        self.pred = pred[ind]
        self.y    = y
        self.softmax = tf.nn.softmax(self.pred, dim=0)
        self.cost1 = tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=y, dim=0)
        self.cost += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=y, dim=0))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost, global_step=self.global_step)
        #self.accuracy = tf.metrics.accuracy(labels=y, predictions=pred)

def gcn_weights_biases(size):
    weights = {}
    biases  = {}
    for i in range(len(size)-2):
        name_w = 'h' + str(i)
        name_b = 'b' + str(i)
        weights[name_w] = tf.Variable(tf.random_normal([size[i], size[i+1]]))
        biases[name_b]  = tf.Variable(tf.random_normal([size[i+1]]))
    i += 1
    name_w = 'h' + str(i)
    name_b = 'b' + str(i)
    weights[name_w] = tf.Variable(tf.random_normal([size[i], size[i+1]]))
    biases[name_b]  = tf.Variable(tf.random_normal([size[i+1]]))


    return(weights, biases)

def gcn_weights_biases_skip(size):
    weights = {}
    biases  = {}
    n_input = size[0]
    for i in range(len(size)-2):
        name_w = 'h' + str(i)
        name_b = 'b' + str(i)
        weights[name_w] = tf.Variable(tf.random_normal([size[i] + n_input, size[i+1]]))
        biases[name_b]  = tf.Variable(tf.random_normal([size[i+1]]))
    i += 1
    name_w = 'h' + str(i)
    name_b = 'b' + str(i)
    weights[name_w] = tf.Variable(tf.random_normal([size[i] + n_input, size[i+1]]))
    biases[name_b]  = tf.Variable(tf.random_normal([size[i+1]]))


    return(weights, biases)





def gcn_train_model(sess, model, data, placeholders, node_list, train_inds, training_epochs=30, display_step=1):
    features, routes, _, _, _, _, ws, optimal, _, _ = data
    x, y, A_tf, global_step, visited = placeholders

    keys       = list(features.keys())
    tot_ex     = len(features.keys())
    epoch_res  = []

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
                batch_y  = routes[keys[ind]][in_ind]
                W        = copy.copy(ws[keys[ind]][in_ind])
                w        = W + np.eye(num_nodes)
                Dinv = np.sqrt(np.diag(1/w.sum(axis=1)))
                A_feed = np.dot(Dinv, w).dot(Dinv)

                for i in range(batch_x.shape[2]):
                    x_feed = batch_x[:,:,i]
                    x_feed = x_feed / x_feed.sum(0)
                    #x_feed = x_feed/num_nodes
                    y_feed = np.zeros([num_nodes, 1])
                    y_feed[batch_y[i]] = 1


                    feed_dict = {}
                    feed_dict[A_tf] = A_feed
                    feed_dict[x] = x_feed
                    feed_dict[y] = y_feed

                    _, pred, y_t, softmax, c1, c = sess.run([model.optimizer, model.pred, model.y,
                                                    model.softmax, model.cost1, model.cost], 
                                                        feed_dict=feed_dict)
                    
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



def score_gcn_class(sess, model, data, placeholders, node_list, test_inds):
    features, routes, _, _, _, _, ws, optimal, _, _ = data
    x, y, A_tf, global_step, visited = placeholders
    keys       = list(features.keys())
    tot_ex     = len(features.keys())

    test_inds = np.sort(test_inds)
    solved = 0
    unsolved = 0
    unsolved_key = []
    solved_key = []
    solved_cost = {}
    optimal_cost = {}
    greedy_cost  = {}
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
            opt_cost = optimal[keys[ind]][in_ind]
            W        = copy.copy(ws[keys[ind]][in_ind])
            w        = W + np.eye(num_nodes)
            Dinv     = np.sqrt(np.diag(1/w.sum(axis=1)))
            A_feed   = np.dot(Dinv, w).dot(Dinv)

            greedy   = greedy_tsp(num_nodes, W, in_ind)

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
            feed_dict[A_tf] = A_feed

            cost = 0
            steps = 0
            cur_node = start_node
            solved_bool = False
            while steps < num_nodes:
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
                    solved += 1
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
                unsolved += 1
                unsolved_key.append(keys[ind])


    subopt = {}
    subopt['unsolved'] = unsolved
    subopt['solved'] = solved

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
            if len(greedy_cost[num_nodes][ind]) > 0:
                gc1.append(min(greedy_cost[num_nodes][ind]))
                sc1.append(min(solved_cost[num_nodes][ind]))
                oc1.append(min(optimal_cost[num_nodes][ind]))

        #print(np.mean(solved_cost1 - optimal_cost1)/optimal_cost1)
        greedy_cost1 = np.array(greedy_cost1)
        optimal_cost1 = np.array(optimal_cost1)
        solved_cost1 = np.array(solved_cost1)
        gc1 = np.array(gc1)
        oc1 = np.array(oc1)
        sc1 = np.array(sc1)
        subopt[num_nodes] = []


        subopt[num_nodes].append(np.mean((np.array(greedy_cost1)-np.array(optimal_cost1))/np.array(optimal_cost1)))
        subopt[num_nodes].append(np.mean((np.array(gc1)-np.array(oc1))/np.array(oc1)))

        subopt[num_nodes].append(np.mean((solved_cost1-optimal_cost1)/optimal_cost1))
        subopt[num_nodes].append(np.mean((sc1-oc1)/oc1))
        #print(np.mean((sc1 - oc1)/oc1))
    return(subopt)










