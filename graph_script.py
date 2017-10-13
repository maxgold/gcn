## script to run experiments with GCN

from __future__ import print_function
import tensorflow as tf
import numpy as np
import json
import time
from graph_layer import *
from gen_tsp import gen_tsp_data, gen_tsp_data_cycle_linked
import pickle


### ALL


field_size = [1, 1, 1]
skip = True
sizes = [6, 6, 6, 6, 1]
num_layers = len(field_size)
#data = gen_tsp_data_cycle_linked(node_list, num_ex, field_size, num_layers, edges)


n_input = 6
n_out  = 1

if skip:
    weights, biases = construct_weights_biases_skip(sizes)
else:
    weights, biases = construct_weights_biases(sizes)

starter_learning_rate = .001
decay_step = int((200*8)**2/3)
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                            decay_step, .997, staircase=True)

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

# tf Graph input

training_epochs = 30
batch_size      = 1
display_step    = 1

# Training cycle

complete_res = pickle.load(open('complete_results.p', 'r'))

#for num_nodes in range(3,20):
	
num_nodes = 12
node_list = [num_nodes]
edges = {}
for node in node_list:
	edges[node] = 2*node
num_train = 250
num_test  = 100
data_train = gen_tsp_data(node_list, num_train, field_size, num_layers)
data_test  = gen_tsp_data(node_list, num_test, field_size, num_layers)
train_inds = range(num_train)[:int(.9*num_train)]
test_inds  = range(num_train)[int(.9*num_train):]

train_model_class(sess, model, data_train, placeholders, field_size, train_inds, 
                 test_inds, node_list, training_epochs, batch_size, display_step)


test_inds = range(num_test)
res = score_model_class(sess, model, data_test, placeholders, field_size, node_list, test_inds)
print(res)
complete_res[num_nodes] = res
f = open('complete_results.p', 'wb')
pickle.dump(complete_res, f)
f.close()















### LOW

node_list = [3,6,9,12]
edges = {}
for node in node_list:
	edges[node] = 2*node
num_ex = 250
file_name = 'graphs_20_ex200.json'
field_size = [1, 1, 1]
skip = True
sizes = [6, 6, 6, 6, 1]
num_layers = len(field_size)
data = gen_tsp_data(node_list, num_ex, field_size, num_layers)
	#data = gen_tsp_data_cycle_linked(node_list, num_ex, field_size, num_layers, edges)

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

starter_learning_rate = .001

decay_step = int(train_size*np.mean(np.array(node_list))**2)
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                            decay_step, .995, staircase=True)

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

# tf Graph input

training_epochs = 30
batch_size      = 1
display_step    = 1

# Training cycle

train_model_class(sess, model, data, placeholders, field_size, train_inds, 
                 test_inds, node_list, training_epochs, batch_size, display_step)


node_list = [4, 5, 6, 7, 8, 9, 10, 11, 12]
edges = {}
for node in node_list:
	edges[node] = 2*node
num_ex = 50
file_name = 'graphs_20_ex200.json'
field_size = [1, 1, 1]
skip = True
sizes = [6, 6, 6, 6, 1]
num_layers = len(field_size)
data = gen_tsp_data(node_list, num_ex, field_size, num_layers)
#data_test = gen_tsp_data_cycle_linked(node_list, num_ex, field_size, num_layers, edges)
test_inds = range(num_ex*len(node_list))


res = score_model_class(sess, model, data_test, placeholders, field_size, node_list, test_inds)
print(res)
complete_res = {}
for node in node_list:
	complete_res[node] = res[node]

f = pickle.load(open('complete_results.p', 'r'))
pickle.dump(complete_res, f)

#### HIGH


node_list = [14, 16, 18]
edges = {}
for node in node_list:
	edges[node] = 2*node
num_ex = 250
file_name = 'graphs_20_ex200.json'
field_size = [1, 1, 1]
skip = True
sizes = [6, 6, 6, 6, 1]
num_layers = len(field_size)
data = gen_tsp_data(node_list, num_ex, field_size, num_layers)
	#data = gen_tsp_data_cycle_linked(node_list, num_ex, field_size, num_layers, edges)

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

starter_learning_rate = .002
decay_step = int(train_size*np.mean(np.array(node_list))**2/3)
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                            decay_step, .985, staircase=True)

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

# tf Graph input

training_epochs = 30
batch_size      = 1
display_step    = 1

# Training cycle

train_model_class(sess, model, data, placeholders, field_size, train_inds, 
                 test_inds, node_list, training_epochs, batch_size, display_step)


node_list = [13, 14, 15, 16, 17, 18]
edges = {}
for node in node_list:
	edges[node] = 2*node
num_ex = 50
file_name = 'graphs_20_ex200.json'
field_size = [1, 1, 1]
skip = True
sizes = [6, 6, 6, 6, 1]
num_layers = len(field_size)
data_test = gen_tsp_data(node_list, num_ex, field_size, num_layers)
#data_test = gen_tsp_data_cycle_linked(node_list, num_ex, field_size, num_layers, edges)
test_inds = range(num_ex*len(node_list))


res = score_model_class(sess, model, data_test, placeholders, field_size, node_list, test_inds)
print(res)
complete_res = pickle.load(open('complete_results.p', 'r'))
for node in node_list:
	complete_res[node] = res[node]

f = open('complete_results.p', 'wb')
pickle.dump(complete_res, f)
f.close()











training_epochs = 12
batch_size      = 1
display_step    = 1

# Training cycle
skip = True

sizes = [n_input, 6, 6, n_out]
if skip:
	weights, biases = construct_weights_biases_skip(sizes)
else:
	weights, biases = construct_weights_biases(sizes)
params['weights'] = weights
params['biases'] = biases
params['loss'] = 'l2'
params['skip'] = skip


sess = tf.Session()

model_reg = graph_conv_net_reg(params)
y1 = tf.placeholder("float")
y2 = tf.placeholder("float")
index_tf = tf.placeholder('int32')
placeholders = [x, y1, y2, P1_tf, P2_tf, Ann_tf, F_tf, global_step, index_tf]
model_reg.create_layers(x,y1, y2, P1_tf,P2_tf,Ann_tf, F_tf, index_tf)
init = tf.global_variables_initializer()
sess.run(init)


train_model_reg(sess, model_reg, data, placeholders, field_size, train_inds, 
                 training_epochs, batch_size, display_step)

res = score_model_reg(sess, model_reg, data, placeholders, field_size, node_list, test_inds)



# RESET THE LEARNING RATE
# model.global_step = tf.Variable(0, trainable=False)
# model.learning_rate = tf.train.exponential_decay(starter_learning_rate, model.global_step,
#                                           decay_step, 0.97, staircase=True)

# init = tf.variables_initializer([model.global_step])
# sess.run(init)





