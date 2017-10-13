## functions to solve TSP with A*
## Also contains a script to do the leapfrogging heuristic 
## experiments

from heapq import heappush, heappop
import heapq
from sys import maxint
import numpy as np
import networkx as nx
from gen_tsp import nn_mats_from_adj_fieldsize_beta, construct_adj_fieldsize
from gen_tsp import google_or_cost, gen_tsp_data, gen_tsp_data_cycle_linked
from gen_tsp import get_dist_to_go, get_ranking_targets
from graph_layer import *
import tensorflow as tf
import time
import pickle
from tsp_utils import edges_from_mat
import random


class State(object):
    def __init__(self, nodes, t_val, h_val):
        self.nodes = nodes
        self.t_val = t_val
        self.h_val = h_val

    def __cmp__(self, other):
        return cmp(self.h_val, other.h_val)

    def __eq__(self, other):
        # and self.nodes != REMOVED
        return self.nodes == other.nodes


class Graph(object):
    def __init__(self, n, start, W):
        self.nodes = range(n)
        self.W     = W
        self.num_nodes = n
        self.start = start
        self.nx_graph = nx.from_numpy_matrix(self.W)

    def neighbors(self, visited):
        res = []
        cur_node = visited[-1]
        if len(visited) < self.num_nodes:
            unvisited = diff(self.nodes, visited)
            for y in unvisited:
                if self.W[cur_node, y] != 0:
                    res.append(visited + [y])
        else:
            if self.W[cur_node, self.start]!=0:
                res.append(visited + [self.start])
        return(res)
    def heuristic(self, visited):
        if len(visited) < self.num_nodes:
            unvisited = np.array(diff(self.nodes, visited))
            new_W = self.W[unvisited[:,None], unvisited]
            nxg = nx.from_numpy_matrix(new_W)
            t = nx.minimum_spanning_tree(nxg)
            return(t.size(weight='weight'))
        else:
            return(0)
    def heuristic_model(self, sess, model, placeholders, visited_l, cur_node, next_cand):
        x, y, P1_tf, P2_tf, Ann_tf, F_tf, global_step, visited = placeholders
        visited_n = np.zeros(self.num_nodes)
        visited_n[visited_l] = 1
        W1 = self.W
        A  = (W1 > 0).astype(int)
        W  = W1/self.num_nodes
        goal_node = self.start
        start_node = self.start
        adj0 = construct_adj_fieldsize(A, W, field_size, num_layers)
        P1_feed, P2_feed, Ann_feed, F_feed = nn_mats_from_adj_fieldsize_beta(adj0, field_size, num_layers)

        state0 = np.zeros([self.num_nodes, 3])
        state0[cur_node, 0] = 1
        state0[goal_node, 1] = 1
        state0[:, 2] = visited_n
        goal0 = np.zeros([self.num_nodes, 3])
        goal0[goal_node, 0] = 1
        goal0[goal_node, 1] = 1
        goal0[:, 2] = 1
        x_feed = np.c_[state0, goal0]
        x_feed = x_feed / x_feed.sum(0)
        
        feed_dict = {}
        for layer in range(num_layers):
            feed_dict[P1_tf[layer]]  = P1_feed[layer]
            feed_dict[P2_tf[layer]]  = P2_feed[layer]
            feed_dict[Ann_tf[layer]] = Ann_feed[layer]
            feed_dict[F_tf[layer]]   = F_feed[layer]

        feed_dict[x] = x_feed
        pred = sess.run(model.softmax, feed_dict=feed_dict)
        h    = ((self.num_nodes - np.sum(visited_n))*(1-pred))[next_cand]

        return(h[0])


def get_inds(open_list, new_state):
    indices = [i for i, x in enumerate(open_list) if x == new_state]
    return(indices)

def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]




def model_imitation_cost(model, W0, start_node):
    A  = (W0 > 0).astype(int)
    w  = W0/num_nodes
    goal_node = start_node
    start_node = start_node
    adj0 = construct_adj_fieldsize(A, w, field_size, num_layers)
    P1_feed, P2_feed, Ann_feed, F_feed = nn_mats_from_adj_fieldsize_beta(adj0, field_size, num_layers)

    visited = np.zeros(num_nodes)
    visited[start_node] = 1
    visited = visited.astype(int)
    route = [start_node]

    state0 = np.zeros([num_nodes, 3])
    state0[start_node, 0] = 1
    state0[goal_node, 1] = 1
    state0[:, 2] = visited
    goal0 = np.zeros([num_nodes, 3])
    goal0[goal_node, 0] = 1
    goal0[goal_node, 1] = 1
    goal0[:, 2] = 1
    x_feed = np.c_[state0, goal0]
    x_feed = x_feed / x_feed.sum(0)
    
    feed_dict = {}
    for layer in range(num_layers):
        feed_dict[P1_tf[layer]]  = P1_feed[layer]
        feed_dict[P2_tf[layer]]  = P2_feed[layer]
        feed_dict[Ann_tf[layer]] = Ann_feed[layer]
        feed_dict[F_tf[layer]]   = F_feed[layer]

    feed_dict[x] = x_feed

    cost = 0
    steps = 0
    cur_node = start_node
    solved_bool = False
    while steps < num_nodes:
        feed_dict[x] = x_feed
        visited_loc = list(np.where(visited==1)[0])
        visited_loc = [a for a in visited_loc if a != start_node]
        row = W0[cur_node]

        softmax = sess.run([model.softmax], feed_dict=feed_dict)
        softmax = softmax[0][:,0]
        #print(np.max(softmax))
        steps += 1
        cand = [n for n in np.argsort(-softmax) if (n not in visited_loc) and row[n]!=0]
        if len(cand) > 0:
            next_node = cand[0]
        else:
            break
        if W0[cur_node, next_node] == 0:
            print(ind, in_ind)
            print('ahhh')
        #print(next_node)
        state0 = np.zeros([num_nodes, 3])
        state0[next_node, 0] = 1
        state0[goal_node, 1] = 1
        visited[next_node] = 1
        state0[:, 2] = visited
        cost += W0[cur_node, next_node]
        cur_node = next_node

        if (next_node == start_node) & (np.sum(visited)==num_nodes):
            solved_bool = True
            break
        x_feed = np.c_[state0, goal0]
        x_feed = x_feed / x_feed.sum(0)
        route.append(next_node)

    if solved_bool:
        return(cost, route)
    else:
        return(-1, [])

def my_astar(G, start, heuristic = False, scale=1, mode='MST', sess=None, model = None, placeholders=None):
    start_t = time.time()
    G.start   = start
    num_nodes = G.num_nodes
    cur_node  = start
    # second node should be value of minimum spanning tree
    start_state    = State([cur_node], 0, 0)
    open_list      = []
    closed_list    = []
    open_storage   = []
    closed_storage = []
    open_set       = set([])
    closed_set     = set([])
    seen_states = {}
    seen_states[(cur_node)] = [0, 0]

    heapq.heappush(open_list, start_state)
    open_set.add(tuple(start_state.nodes))
    visited = [start]
    unfinished = True

    explored = 0
    while unfinished:
        if explored % 100 == 0 and explored > 0:
            None
            #print(explored)
        explored += 1
        cur_state = heapq.heappop(open_list)
        #print(cur_state.nodes)
        open_set.remove(tuple(cur_state.nodes))
        visited   = cur_state.nodes
        cur_node  = cur_state.nodes[-1]
        if (len(visited) == (num_nodes + 1)) & (cur_node == start):
            unfinished = False
            break
        for new_state in G.neighbors(visited):
            if tuple(new_state) in open_set.union(closed_set):
                prev_tour_cost, prev_h_cost = seen_states[new_state]
            else:
                prev_tour_cost, prev_h_cost = np.inf, np.inf

            new_node = new_state[-1]
            successor_tour_cost = cur_state.t_val + G.W[cur_node, new_node]
            if heuristic and mode == 'MST':
                h = scale*G.heuristic(new_state)
            elif heuristic and mode == 'NN':
                h = scale*G.heuristic_model(sess, model, placeholders, visited, cur_node, new_node)
            else:
                h = 0
            # the heap sorts by successor_h_cost
            successor_h_cost    = successor_tour_cost + h
            #successor_h_cost    = h
            new_state_cost = State(new_state, successor_tour_cost, successor_h_cost)
            if tuple(new_state_cost.nodes) in open_set:
                if prev_h_cost <= successor_h_cost:
                    # this should break to neighbor loop
                    break
                else:
                    inds = get_inds(open_list, new_state_cost)
                    assert(len(inds) == 1)
                    open_list.remove(new_state_cost)
                    heapq.heappush(open_list, new_state_cost)
                    seen_states[tuple(new_state)] = [successor_tour_cost, successor_h_cost]
            elif tuple(new_state_cost.nodes) in closed_set:
                if prev_h_cost <= successor_h_cost:
                    # this should break to the neighbor loop    
                    break
                else:
                    inds = get_inds(closed_list, new_state_cost)
                    for ind in inds:
                        closed_list.remove(new_state_cost)
                    heapq.heappush(closed_list, new_state_cost)
                    seen_states[tuple(new_state)] = [successor_tour_cost, successor_h_cost]                       
            else:
                seen_states[tuple(new_state)] = [successor_tour_cost, successor_h_cost]
                heapq.heappush(open_list, new_state_cost)
                open_set.add(tuple(new_state_cost.nodes))   

        heappush(closed_list, cur_state)
        closed_set.add(tuple(cur_state.nodes))


    end = time.time()
    #print(end-start_t)
    #print(cur_state.nodes)
    #print(cur_state.t_val)
    if unfinished:
        return(-1, -1, -1)
    else:
        return(cur_state.nodes, cur_state.t_val, explored)

def cycle_linked_matrix(num_nodes, num_new_edges):
    matrix = np.zeros((num_nodes, num_nodes))
    cycle_weights = np.random.rand(num_nodes)
    cycle_cost = np.sum(cycle_weights)
    for i in range(num_nodes):
      matrix[i, (i+1)%num_nodes] = cycle_weights[i]
      matrix[(i+1)%num_nodes, i] = cycle_weights[i]
    
    cycle_edges = edges_from_mat(matrix)
    t = np.ones((num_nodes, num_nodes))
    np.fill_diagonal(t, 0)
    all_edges = np.array(edges_from_mat(t))
    all_edges = all_edges[all_edges[:,0] < all_edges[:,1]]
    all_edges = list(map(tuple, all_edges))
    new_edges = [x for x in all_edges if x not in cycle_edges]
    random.shuffle(new_edges)
    new_edges = np.array(new_edges)
    num_new_edges = min(len(new_edges), num_new_edges)
    for i in range(num_new_edges):
      val = np.random.rand(1)
      matrix[new_edges[i,0],new_edges[i,1]] = val
      matrix[new_edges[i,1],new_edges[i,0]] = val
    return(matrix)


def astar_assignment(sess, model, G, start_node, placeholders, mode='MST'):
    if mode == 'MST':
        opt_route, opt_val, explored = my_astar(G, start_node, heuristic=True,scale=1,mode='MST')
    else:
        opt_route, opt_val, explored = my_astar(G, start_node, heuristic=True, scale=.5, 
                                    mode='NN', sess=sess, model=model, 
                                        placeholders=placeholders)
    return(opt_route, opt_val, explored)

def gen_tsp_astar(sess, model_init, num_nodes, num_inds, field_size, num_layers, mode='NN'):
    features  = {}
    routes    = {}
    P1        = {}
    P2        = {}
    A_nn      = {}
    F         = {}
    adj       = {}
    ws        = {}
    optimal   = {}
    dist_left = {}
    rankings  = {}
    explored_nodes = []

    for ind in range(num_inds):
        if ind % 1 == 0:
            print(ind)
        key1 = (ind, num_nodes)
        features[key1]  = {}
        routes[key1]    = {}
        adj[key1]       = {}
        P1[key1]        = {}
        P2[key1]        = {}
        A_nn[key1]      = {}
        F[key1]         = {}
        ws[key1]        = {}
        optimal[key1]   = {}
        dist_left[key1] = {}
        rankings[key1]  = {}
        W = np.random.rand(num_nodes, num_nodes)
        np.fill_diagonal(W, 0)
        W = (W + W.T)/2
        G = Graph(num_nodes, 0, W)
        A = (W > 0).astype(int)


        #W = W1 / np.mean(W1.sum(0))
        W1 = W / num_nodes
        min_opt = np.inf
        min_route = []
        for start_node in range(num_nodes):
            route, opt_val, explored = astar_assignment(sess, model_init, G, 
                                start_node, placeholders, mode)
            print(explored)
            explored_nodes.append(explored)
            if opt_val < min_opt and opt_val != -1:
                min_opt   = opt_val
                min_route = route
                
        for start_node in min_route[:-1]:
            feature0, route0 = feature_from_route2(min_route, num_nodes, start_node)
            adj0       = construct_adj_fieldsize(A, W1, field_size, num_layers)
            P1_0, P2_0, A_nn0, F_0 = nn_mats_from_adj_fieldsize_beta(adj0, field_size, num_layers)
            features[key1][start_node] = feature0
            routes[key1][start_node] = route0
            adj[key1][start_node] = adj0
            P1[key1][start_node] = P1_0
            P2[key1][start_node] = P2_0
            A_nn[key1][start_node] = A_nn0
            F[key1][start_node]    = F_0
            ws[key1][start_node] = W
            optimal[key1][start_node] = min_opt
            dist_left[key1][start_node] = get_dist_to_go(start_node, route0, opt_val, W)
            rankings[key1][start_node]  = get_ranking_targets(start_node, route0, num_nodes)
            

    return(features, routes, P1, P2, A_nn, F, ws, optimal, dist_left, rankings, explored_nodes)


def gen_tsp_astar_cycle_linked(sess, model_init, num_nodes, num_inds, field_size, num_layers, edges, mode='MST'):
    features  = {}
    routes    = {}
    P1        = {}
    P2        = {}
    A_nn      = {}
    F         = {}
    adj       = {}
    ws        = {}
    optimal   = {}
    dist_left = {}
    rankings  = {}
    explored_nodes = []
    num_new_edges = edges[num_nodes]


    for ind in range(num_inds):
        if ind % 1 == 0:
            print(ind)
        key1 = (ind, num_nodes)
        features[key1]  = {}
        routes[key1]    = {}
        adj[key1]       = {}
        P1[key1]        = {}
        P2[key1]        = {}
        A_nn[key1]      = {}
        F[key1]         = {}
        ws[key1]        = {}
        optimal[key1]   = {}
        dist_left[key1] = {}
        rankings[key1]  = {}
        

        W = cycle_linked_matrix(num_nodes, num_new_edges)
        G = Graph(num_nodes, 0, W)
        A = (W > 0).astype(int)


        #W = W1 / np.mean(W1.sum(0))
        W1 = W / num_nodes
        min_opt = np.inf
        min_route = []
        for start_node in range(num_nodes):
            route, opt_val, explored = astar_assignment(sess, model_init, G, 
                                start_node, placeholders, mode)
            explored_nodes.append(explored)
            print(explored)
            if opt_val < min_opt and opt_val != -1:
                min_opt = opt_val
                min_route = route
                
        for start_node in min_route[:-1]:
            feature0, route0 = feature_from_route2(min_route, num_nodes, start_node)
            adj0       = construct_adj_fieldsize(A, W1, field_size, num_layers)
            P1_0, P2_0, A_nn0, F_0 = nn_mats_from_adj_fieldsize_beta(adj0, field_size, num_layers)
            features[key1][start_node] = feature0
            routes[key1][start_node] = route0
            adj[key1][start_node] = adj0
            P1[key1][start_node] = P1_0
            P2[key1][start_node] = P2_0
            A_nn[key1][start_node] = A_nn0
            F[key1][start_node]    = F_0
            ws[key1][start_node] = W
            optimal[key1][start_node] = min_opt
            dist_left[key1][start_node] = get_dist_to_go(start_node, route0, opt_val, W)
            rankings[key1][start_node]  = get_ranking_targets(start_node, route0, num_nodes)
            

    return(features, routes, P1, P2, A_nn, F, ws, optimal, dist_left, rankings, explored_nodes)



def feature_from_route2(route, num_nodes, start_node):
  ind = [i for (i,x) in zip(range(num_nodes+1),route) if x == start_node][0]
  new_route = route[ind:-1]
  new_route.extend(route[:ind])

  features  = np.zeros([num_nodes,6,0])
  i         = 1
  goal_node = start_node
  visited   = np.zeros(num_nodes)
  visited[start_node] = 1
  goal = np.zeros([num_nodes, 3])
  goal[goal_node, 0] = 1
  goal[goal_node, 1] = 1
  goal[:, 2] = 1
  state0 = np.zeros([num_nodes, 3])
  state0[start_node, 0] = 1
  state0[goal_node, 1]  = 1
  state0[:, 2] = visited
  feature0 = np.c_[state0, goal][:,:,None]
  features = np.concatenate((features,feature0), axis=2)

  final_route = [start_node]
  while np.sum(visited)!=num_nodes:
    cur_node = new_route[i]
    final_route.append(cur_node)
    state0 = np.zeros([num_nodes, 3])
    state0[cur_node, 0] = 1
    state0[goal_node, 1]  = 1
    visited[cur_node] = 1
    state0[:, 2] = visited
    feature0 = np.c_[state0, goal][:,:,None]
    features = np.concatenate((features,feature0), axis=2)
    i += 1
  final_route.append(start_node)
  assert(len(final_route) == num_nodes+1)
  return(features, final_route[1:])

def astar_comparison(f, sess, model, placeholders, node_list, num_ex, mst=True):
    astar_res = pickle.load(open(f, 'r'))
    for num_nodes in node_list:
        nn_vals      = []
        nn_explored  = []
        mst_vals     = []
        mst_explored = []
        il_vals      = []
        google_vals  = []
        bad = False
        for i in range(num_ex):
            print(i)
            W = np.random.rand(num_nodes, num_nodes)
            np.fill_diagonal(W, 0)
            W = (W + W.T)/2
            G = Graph(num_nodes, 0, W)
            for start in range(num_nodes):
                nn_route, nn_val, nn_n = my_astar(G, start, heuristic=True, scale=.5, 
                                        mode='NN', sess=sess, model=model, 
                                            placeholders=placeholders)
                if mst:
                    mst_route, mst_val, mst_n = my_astar(G, start, heuristic=True, scale=1, mode='MST')
                #il_val = model_imitation_cost(model, W, start)[0]
                google_val = google_or_cost(W, start)

                nn_vals.append(nn_val)
                nn_explored.append(nn_n)
                if mst:
                    mst_vals.append(mst_val)
                    mst_explored.append(mst_n)
                #il_vals.append(il_val)
                google_vals.append(google_val)


        astar_res[num_nodes] = {}
        astar_res[num_nodes]['nn_explored'] = nn_explored
        astar_res[num_nodes]['nn_vals'] = nn_vals
        if mst:
            astar_res[num_nodes]['mst_vals'] = mst_vals
            astar_res[num_nodes]['mst_explored'] = mst_explored
        #astar_res[num_nodes]['il_vals'] = il_vals
        astar_res[num_nodes]['google_vals'] = google_vals
        f = open(f, 'wb')
        pickle.dump(astar_res, f)
        f.close()




if __name__ == 'main':

    early_stop = {}
    early_stop[4]  = .3
    early_stop[5]  = .33
    early_stop[6]  = .42
    early_stop[7]  = .42
    early_stop[8]  = .45
    early_stop[9]  = .45
    early_stop[10] = .5
    early_stop[11] = .45
    early_stop[12] = .5
    early_stop[13] = .5
    early_stop[14] = .53
    early_stop[15] = .56
    early_stop[16] = .6



    num_train  = 100
    # testing will be special with MST
    num_test   = 10
    field_size = [1, 1, 1]
    skip       =  True
    sizes      = [6, 6, 6, 6, 1]
    num_layers = 3

    tot_ex = num_train
    train_size = int(.9*tot_ex)

    inds       = np.random.permutation(tot_ex)
    train_inds = inds[:train_size]
    test_inds  = inds[train_size:]


    n_input = 6
    n_out  = 1

    if skip:
        weights, biases = construct_weights_biases_skip(sizes)
    else:
        weights, biases = construct_weights_biases(sizes)

    starter_learning_rate = .0015
    decay_step = int(train_size*4)**2/3
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

    training_epochs = 15
    batch_size      = 1
    display_step    = 1


    cycle = True


    data = {}

    init_nodes = 4
    node_list = [4]
    data[init_nodes] = {}

    if not cycle:
        data[init_nodes]['train'] = gen_tsp_data(node_list, num_train, field_size, num_layers)
        data[init_nodes]['test']  = gen_tsp_data(node_list, num_test, field_size, num_layers)
    else:
        edges = {}
        edges[node_list[0]] = node_list[0]
        data[init_nodes]['train'] = gen_tsp_data_cycle_linked(node_list, num_train, field_size, num_layers, edges)
        data[init_nodes]['test']  = gen_tsp_data_cycle_linked(node_list, num_test, field_size, num_layers, edges)

    data_sub = data[init_nodes]['train']
    train_model_class(sess, model, data_sub, placeholders, field_size, train_inds, 
                     test_inds, node_list, training_epochs, batch_size, display_step)


    if cycle:
        f = 'astar_results_cycle.p'
        file2 = 'leapfrog_astar_cycle.p'
    else:
        f = 'astar_results.p'
        file2 = 'leapfrog_astar2.p'

    for num_nodes in range(5,6):
        training_epochs = 15
        batch_size      = 1
        display_step    = 1

        res_dic = pickle.load(open(file2, 'r'))

        node_list = [num_nodes]
        data[num_nodes] = {}
        num_train = 100
        num_test  = 25
        
        if not cycle:
            temp_data = gen_tsp_astar(sess, model, num_nodes, num_train, field_size, num_layers, mode='NN')
        else:
            edges = {}
            edges[num_nodes] = 2*num_nodes
            temp_data = gen_tsp_astar_cycle_linked(sess, model, num_nodes, num_train, field_size, num_layers, edges, mode='NN')
        data[num_nodes]['train'] = temp_data[:-1]
        NN_explored = np.array(temp_data[-1])
        if num_nodes <= 8:
            if not cycle:
                temp_data = gen_tsp_astar(sess, model, num_nodes, num_test, field_size, num_layers, mode='MST')
            else:
                edges = {}
                edges[num_nodes] = [num_nodes]
                temp_data = gen_tsp_astar_cycle_linked(sess, model, num_nodes, num_test, field_size, num_layers, edges, mode='MST')
            data[num_nodes]['mst'] = temp_data[:-1]
            MST_explored           = temp_data[-1]
        if not cycle:
            temp_data   = gen_tsp_data(node_list, num_test, field_size, num_layers)
        else:
            edges = {}
            edges[num_nodes] = 2*num_nodes
            temp_data   = gen_tsp_data_cycle_linked(node_list, num_test, field_size, num_layers, edges)
        data[num_nodes]['google'] = temp_data

        data_train = data[num_nodes]['train']
        train_inds = range(int(.9*num_train))
        test_inds = range(num_test)

        early_ll = early_stop[num_nodes]
        train_model_class(sess, model, data_train, placeholders, field_size, train_inds, 
                         test_inds, node_list, training_epochs, batch_size, display_step, early_ll)

        res = {}
        if num_nodes <= 8:
            data_test  = data[num_nodes]['mst']
            res_mst = score_model_class(sess, model, data_test , placeholders, field_size, node_list, test_inds)
            res_mst['mean_nn_explored'] = np.mean(NN_explored)
            res_mst['sd_nn_explored'] = np.std(NN_explored)
            res_mst['sd_mst_explored'] = np.std(MST_explored)
            res_mst['mean_mst_explored'] = np.mean(MST_explored)
            res['mst'] = res_mst
            astar_comparison(f, sess, model, placeholders, node_list, 50, mst=True)
        else:
            astar_comparison(f, sess, model, placeholders, node_list, 50, mst=False)
        data_test  = data[num_nodes]['google']
        res_google = score_model_class(sess, model, data_test, placeholders, field_size, node_list, test_inds)   
        res['google'] = res_google
        print(res)
        res_dic[num_nodes] = res
        ft = open(file2, 'wb')
        pickle.dump(res_dic, ft)
        ft.close()




    astar_res = pickle.load(open(f,'r'))
    node = 11

    nn_explored  = astar_res[node]['nn_explored']
    nn_vals      = astar_res[node]['nn_vals']
    if node <= 8:
        mst_vals     = astar_res[node]['mst_vals']
        mst_explored = astar_res[node]['mst_explored']
    google_vals  = astar_res[node]['google_vals']

    nn_vals, nn_explored = np.array(nn_vals), np.array(nn_explored)
    mst_vals, mst_explored = np.array(mst_vals), np.array(mst_explored)
    google_vals = np.array(google_vals)

    if node <= 8:
        print('MST to NN vals')
        print(np.mean((nn_vals-mst_vals)/mst_vals))
        print('MST to NN explored')
        print(np.mean((mst_explored-nn_explored)/nn_explored))
        print('Google to opt')
        print(np.mean((google_vals-mst_vals)/mst_vals))
    print('NN to google')
    print(np.mean((nn_vals-google_vals)/google_vals))
    num_ex = int(len(nn_vals)/node)
    nn = np.min(nn_vals.reshape(num_ex,node), axis=1)
    if node <= 8:
        mst = np.min(mst_vals.reshape(num_ex,node), axis=1)
    goog = np.min(google_vals.reshape(num_ex,node), axis=1)
    print('MIN')
    if node <= 8:
        print('MST to NN vals')
        print(np.mean((nn-mst)/mst))
        print('Google to opt')
        print(np.mean((goog-mst)/mst))
    print('NN to Google')
    print(np.mean((nn-goog)/goog))



























