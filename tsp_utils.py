## Utility functions to help generate TSP instances

import numpy as np
import pickle
from copy import copy
import numpy.linalg as nlg
import json
import random

#from tsp_utils import *

class create_adj(object):
  def __init__(self, tsp_size):
    self.scale  = 1000
    self.matrix = np.round(np.random.rand(tsp_size,tsp_size),3)*self.scale
    np.fill_diagonal(self.matrix, 0)
    self.matrix = (self.matrix + self.matrix.T)/2

  def Distance(self, from_node, to_node):
    return self.matrix[from_node][to_node]



def edges_from_mat(matrix):
  edges = zip(np.where(matrix>0)[0], np.where(matrix>0)[1])
  return(edges)


class create_adj_cycle(object):
  def __init__(self, tsp_size, num_new_edges = None):
    if num_new_edges == None:
      num_new_edges = tsp_size
    self.scale = 1000
    cycle_weights = np.random.rand(tsp_size)
    cycle_cost = np.sum(cycle_weights)
    self.matrix = np.zeros((tsp_size, tsp_size))
    for i in range(tsp_size):
      self.matrix[i, (i+1)%tsp_size] = cycle_weights[i]
      self.matrix[(i+1)%tsp_size, i] = cycle_weights[i]
    
    cycle_edges = edges_from_mat(self.matrix)
    t = np.ones((tsp_size, tsp_size))
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
      self.matrix[new_edges[i,0],new_edges[i,1]] = val
      self.matrix[new_edges[i,1],new_edges[i,0]] = val

    self.matrix = np.round(self.matrix, 3) * self.scale
    self.matrix[self.matrix==0] = 1e6
    self.cycle_cost = cycle_cost
    np.fill_diagonal(self.matrix, 0)

  def Distance(self, from_node, to_node):
    return(self.matrix[from_node][to_node])





def distance(x1, y1, x2, y2):
    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return dist

def construct_cycle_feature_route_cw(num_nodes, start_node, goal_node):
  pos_mat  = np.eye(num_nodes)
  pos_mat[start_node, :] = 1
  pos_vec  = np.zeros(num_nodes)
  pos_vec[start_node] = 1

  goal_vec = np.zeros(num_nodes)
  goal_vec[goal_node] = 1
  visited_vec  = np.zeros(num_nodes)
  visited_vec[start_node] = 1
  goal_feature = np.zeros([num_nodes, 3])
  goal_feature[start_node, 0] = 1
  goal_feature[start_node, 1] = 1
  goal_feature[:, 2] = 1
  #feature_tour = np.c_[pos_mat, goal_vec, visited_vec, goal_feature][:,:,None]
  feature_tour = np.c_[pos_vec, goal_vec, visited_vec, goal_feature][:,:,None]
  route0       = np.zeros([num_nodes, 1])
  route0[0]    = (start_node + 1)%num_nodes
  for i in range(1, num_nodes):
    cur_node = (start_node + i) % num_nodes
    pos_mat  = np.eye(num_nodes)
    pos_mat[cur_node, :] = 1
    pos_vec  = np.zeros(num_nodes)
    pos_vec[cur_node] = 1
    goal_vec = np.zeros(num_nodes)
    goal_vec[start_node] = 1
    visited_vec[cur_node] = 1
    #feature0 = np.c_[pos_mat, goal_vec, visited_vec, goal_feature][:,:,None]
    feature0 = np.c_[pos_vec, goal_vec, visited_vec, goal_feature][:,:,None]    
    feature_tour = np.concatenate((feature_tour, feature0), axis=2)
    route0[i]  = (cur_node + 1) % num_nodes

  return(feature_tour, route0)

def construct_cycle_feature_route_ccw(num_nodes, start_node, goal_node):
  pos_mat  = np.eye(num_nodes)
  pos_mat[start_node, :] = 1
  pos_vec  = np.zeros(num_nodes)
  pos_vec[start_node] = 1

  goal_vec = np.zeros(num_nodes)
  goal_vec[goal_node] = 1
  visited_vec  = np.zeros(num_nodes)
  visited_vec[start_node] = 1
  goal_feature = np.zeros([num_nodes, 3])
  goal_feature[start_node, 0] = 1
  goal_feature[start_node, 1] = 1
  goal_feature[:, 2] = 1
  #feature_tour = np.c_[pos_mat, goal_vec, visited_vec, goal_feature][:,:,None]
  feature_tour = np.c_[pos_vec, goal_vec, visited_vec, goal_feature][:,:,None]
  route0       = np.zeros([num_nodes, 1])
  route0[0]    = (start_node - 1)%num_nodes
  for i in range(1, num_nodes):
    cur_node = (start_node - i) % num_nodes
    pos_mat  = np.eye(num_nodes)
    pos_mat[cur_node, :] = 1
    pos_vec  = np.zeros(num_nodes)
    pos_vec[cur_node] = 1
    goal_vec = np.zeros(num_nodes)
    goal_vec[start_node] = 1
    visited_vec[cur_node] = 1
    #feature0 = np.c_[pos_mat, goal_vec, visited_vec, goal_feature][:,:,None]
    feature0 = np.c_[pos_vec, goal_vec, visited_vec, goal_feature][:,:,None]    
    feature_tour = np.concatenate((feature_tour, feature0), axis=2)
    route0[i]  = (cur_node - 1) % num_nodes

  return(feature_tour, route0)

def construct_cycle_adj(num_nodes):
  adj = np.zeros([num_nodes, num_nodes])
  for i in range(num_nodes):
    adj[i, (i+1)%num_nodes] = 1
    adj[(i+1)%num_nodes, i] = 1

  return adj


def construct_cycle_feature_missing(num_nodes, start_node, goal_node, start_missing, end_missing):
  visited = np.zeros(num_nodes)
  visited[start_node] = 1
  goal = np.zeros([num_nodes, 3])
  goal[goal_node, 0] = 1
  goal[goal_node, 1] = 1
  goal[:, 2] = 1
  if start_missing <= end_missing:
    if end_missing - num_nodes != -1:
      goal[(end_missing-num_nodes+1)%num_nodes:,2] = 0
    goal[:start_missing, 2] = 0
  else:
    goal[(end_missing+1)%num_nodes:start_missing,2] = 0
  features = np.zeros([num_nodes,6,0])
  routes   = []
  #routes.append((start_node - 1)%num_nodes)
  cur_node = start_node
  for i in range(0, (start_node - start_missing + num_nodes)%num_nodes):
    cur_node = (start_node - i)%num_nodes
    state0 = np.zeros([num_nodes, 3])
    state0[cur_node, 0] = 1
    state0[goal_node, 1]  = 1
    visited[cur_node] = 1
    state0[:, 2] = visited
    feature0 = np.c_[state0, goal][:,:,None]
    features = np.concatenate((features,feature0), axis=2)
    routes.append((cur_node - 1)%num_nodes)

  for i in range(0, (end_missing - start_missing + num_nodes)%num_nodes):
    cur_node = (start_missing + i)%num_nodes
    state0 = np.zeros([num_nodes, 3])
    state0[cur_node, 0] = 1
    state0[goal_node, 1]  = 1
    visited[cur_node] = 1
    state0[:, 2] = visited
    feature0 = np.c_[state0, goal][:,:,None]
    features = np.concatenate((features,feature0), axis=2)
    routes.append((cur_node + 1)%num_nodes)

  return(features, routes)


def construct_cycle_adj_missing_fieldsize(num_nodes, field_size, num_layers, start_node, start_missing, end_missing):
  adj = np.zeros([num_nodes, num_nodes])
  for i in range(num_nodes):
    adj[i, (i+1)%num_nodes] = 1
    adj[(i+1)%num_nodes, i] = 1

  if end_missing < start_missing:
    adj[end_missing+1:start_missing] = 0
    adj[end_missing, (end_missing+1)%num_nodes] = 0
    adj[start_missing, start_missing-1] = 0
  else:
    adj[end_missing+1:] = 0
    adj[end_missing, (end_missing+1)%num_nodes] = 0
    adj[:start_missing] = 0
    adj[start_missing, (start_missing-1)%num_nodes] = 0

  res = {}
  for i in range(num_layers):
    t = nlg.matrix_power(adj, field_size[i])
    #t[t>0] = 1
    res[i] = t

  return res


def construct_cycle_adj_missing_fieldsize_beta(num_nodes, field_size, num_layers, start_node, start_missing, end_missing):
  adj = np.zeros([num_nodes, num_nodes])
  for i in range(num_nodes):
    adj[i, (i+1)%num_nodes] = 1
    adj[(i+1)%num_nodes, i] = 1

  if end_missing < start_missing:
    adj[end_missing+1:start_missing, :] = 0
    adj[end_missing, (end_missing+1)%num_nodes] = 0
    adj[start_missing, start_missing-1] = 0
  else:
    adj[end_missing+1:, :] = 0
    adj[end_missing, (end_missing+1)%num_nodes] = 0
    adj[:start_missing, :] = 0
    adj[start_missing, (start_missing-1)%num_nodes] = 0

  ### a lot of choices down here
  power_tracker = np.zeros([num_nodes, num_nodes])
  counter = nlg.matrix_power(adj, field_size[0])
  res = {}
  for i in range(num_layers):
    t1 = nlg.matrix_power(adj, field_size[i])
    #t1[power_tracker > 0] = 0
    power_tracker = power_tracker + t1
    t2 = np.zeros([num_nodes,num_nodes])
    t2[t1 > 0] = field_size[i]
    mask = (t2 > 0) & (counter == 0)
    counter[mask] = np.maximum(counter[mask], t2[mask])
    t = np.concatenate((t1[:,:,None],counter[:,:,None]), axis=2)
    #t[t>0] = 1
    res[i] = t

  return res

def construct_cycle_adj_missing(num_nodes, start_node, start_missing, end_missing):
  adj = np.zeros([num_nodes, num_nodes])
  for i in range(num_nodes):
    adj[i, (i+1)%num_nodes] = 1
    adj[(i+1)%num_nodes, i] = 1

  if end_missing < start_missing:
    adj[end_missing+1:start_missing] = 0
    adj[end_missing, (end_missing+1)%num_nodes] = 0
    adj[start_missing, start_missing-1] = 0
  else:
    adj[end_missing+1:] = 0
    adj[end_missing, (end_missing+1)%num_nodes] = 0
    adj[:start_missing] = 0
    adj[start_missing, (start_missing-1)%num_nodes] = 0

  return adj

def construct_cycle_weight(num_nodes, max_weight = 10):
  W = np.zeros([num_nodes, num_nodes])
  for i in range(num_nodes):
    weight = np.random.randint(max_weight)
    W[i, (i+1)%num_nodes] = weight
    W[(i+1)%num_nodes, i] = weight

  return W

def nn_mats_from_adj(A):
  num_edges = np.sum(A > 0)
  num_nodes = A.shape[0]
  edges = np.zeros([0,2])
  R = A.shape[0]
  C = A.shape[1]
  for i in range(C):
      for j in range(R):
          if A[i,j] > 0:
              edges = np.r_[edges, np.array([[i, j]])]

  edges = edges.astype(int)
  P1 = np.zeros([edges.shape[0], num_nodes])
  P2 = np.zeros([edges.shape[0], num_nodes])
  P1[np.arange(edges.shape[0]).astype(int), edges[:,0]] = 1
  P2[np.arange(edges.shape[0]).astype(int), edges[:,1]] = 1

  A_nn = np.zeros([num_nodes, num_edges])
  c = 0
  for edge in edges:
      A_nn[edge[0], c] = 1 # this should be set to W_ij
      c += 1
  return(P1, P2, A_nn)


def nn_mats_from_adj_fieldsize(adj, field_size, num_layers):
  P1d = {}
  P2d = {}
  A_nnd = {}
  Fd     = {}
  num_nodes = adj[0].shape[0]
  for layer in range(num_layers):
    dist = field_size[layer]
    A = adj[layer]
    num_edges = np.sum(A > 0)
    num_nodes = A.shape[0]
    edges = np.zeros([0,2])
    R = A.shape[0]
    C = A.shape[1]
    for i in range(C):
        for j in range(R):
            if A[i,j] > 0:
                edges = np.r_[edges, np.array([[i, j]])]

    edges = edges.astype(int)
    P1 = np.zeros([edges.shape[0], num_nodes])
    P2 = np.zeros([edges.shape[0], num_nodes])
    F  = np.zeros([edges.shape[0], 1])
    P1[np.arange(edges.shape[0]).astype(int), edges[:,0]] = 1
    P2[np.arange(edges.shape[0]).astype(int), edges[:,1]] = 1

    A_nn = np.zeros([num_nodes, num_edges])
    c = 0
    for edge in edges:
      F[c, 0] = A[edge[0], edge[1]]
      #F[c, 1] = A[edge[0], edge[1]] #should be field_size[layer]
      #F[c, 2] = A[edge[0], edge[1], 2] #should be entry of the weight power matrix
      A_nn[edge[0], c] = 1 # this should be set to W_ij
      c += 1
    F = F/num_nodes
    P1d[layer] = P1
    P2d[layer] = P2
    A_nnd[layer] = A_nn
    Fd[layer]    = F

  return(P1d, P2d, A_nnd, Fd)


def nn_mats_from_adj_fieldsize_beta(adj, field_size, num_layers):
  P1d = {}
  P2d = {}
  A_nnd = {}
  Fd     = {}
  num_nodes = adj[0].shape[0]
  for layer in range(num_layers):
    dist = field_size[layer]
    A = adj[layer]
    num_edges = np.sum(A[:,:,0] > 0)
    num_nodes = A.shape[0]
    edges = np.zeros([0,2])
    R = A.shape[0]
    C = A.shape[1]
    for i in range(C):
        for j in range(R):
            if A[i,j, 0] > 0:
                edges = np.r_[edges, np.array([[i, j]])]

    edges = edges.astype(int)
    P1 = np.zeros([edges.shape[0], num_nodes])
    P2 = np.zeros([edges.shape[0], num_nodes])
    F  = np.zeros([edges.shape[0], 2])
    ## TODO try replacing this with the weight
    P1[np.arange(edges.shape[0]).astype(int), edges[:,0]] = 1
    P2[np.arange(edges.shape[0]).astype(int), edges[:,1]] = 1

    A_nn = np.zeros([num_nodes, num_edges])
    c = 0
    for edge in edges:
      F[c, 0] = A[edge[0], edge[1], 0]
      F[c, 1] = A[edge[0], edge[1], 1] #should be field_size[layer]
      #F[c, 2] = A[edge[0], edge[1], 2] #should be entry of the weight power matrix
      A_nn[edge[0], c] = 1 # this should be set to W_ij
      c += 1
    F = F/num_nodes
    P1d[layer] = P1
    P2d[layer] = P2
    A_nnd[layer] = A_nn
    Fd[layer]    = F

  return(P1d, P2d, A_nnd, Fd)

def gen_cycle_data(num_nodes, trials = 100):
  #features   = np.zeros([num_nodes, num_nodes + 5, num_nodes, 0])
  features   = np.zeros([num_nodes, 6, num_nodes, 0])
  weights    = np.zeros([num_nodes, num_nodes, 0])
  adj        = np.zeros([num_nodes, num_nodes, 0])
  routes     = np.zeros([num_nodes, 0])
  P1         = np.zeros([2*num_nodes, num_nodes, 0])
  P2         = np.zeros([2*num_nodes, num_nodes, 0])
  A_nn       = np.zeros([num_nodes, 2*num_nodes, 0])
  for i in range(num_nodes):
    start_node = i
    goal_node  = i
    feature_tour1, route1 = construct_cycle_feature_route_cw(num_nodes, start_node, goal_node)
    feature_tour2, route2 = construct_cycle_feature_route_ccw(num_nodes, start_node, goal_node)
    adj0       = construct_cycle_adj(num_nodes)
    weight0    = construct_cycle_weight(num_nodes)
    P1_0, P2_0, A_nn0 = nn_mats_from_adj(adj0)
    features   = np.concatenate((features, feature_tour1[:,:,:,None]), axis=3)
    features   = np.concatenate((features, feature_tour2[:,:,:,None]), axis=3)
    routes     = np.concatenate((routes, route1), axis=1)
    routes     = np.concatenate((routes, route2), axis=1)
    adj        = np.concatenate((adj, adj0[:,:,None]), axis=2)
    adj        = np.concatenate((adj, adj0[:,:,None]), axis=2)
    weights    = np.concatenate((weights, weight0[:,:,None]), axis=2)
    weights    = np.concatenate((weights, weight0[:,:,None]), axis=2)
    P1         = np.concatenate((P1, P1_0[:,:,None]), axis=2)
    P2         = np.concatenate((P2, P2_0[:,:,None]), axis=2)
    A_nn       = np.concatenate((A_nn, A_nn0[:,:,None]), axis=2)


  return(features, weights, adj, routes, P1, P2, A_nn)

def gen_cycle_data_missing(num_nodes, field_size, num_layers):
  features   = {}
  adj        = {}
  routes     = {}
  P1         = {}
  P2         = {}
  A_nn       = {}
  F          = {}
  for start_missing in range(num_nodes):
    for end_missing in range(num_nodes):
      if (start_missing != end_missing) & (abs((start_missing - end_missing )%num_nodes)!=1):
        if start_missing < end_missing:
          start_vals = np.arange(start_missing,end_missing+1)
        else:
          start_vals = np.arange(start_missing, end_missing + num_nodes + 1) % num_nodes
        for i in start_vals:
          start_node = i
          goal_node  = end_missing
          feature0, route0 = construct_cycle_feature_missing(num_nodes, start_node, goal_node, start_missing, end_missing)
          adj0       = construct_cycle_adj_missing_fieldsize_beta(num_nodes, field_size, num_layers, start_node, start_missing, end_missing)
          P1_0, P2_0, A_nn0, F_0 = nn_mats_from_adj_fieldsize_beta(adj0, field_size, num_layers)
          features[(start_missing, end_missing, i)] = feature0
          routes[(start_missing, end_missing, i)] = route0
          adj[(start_missing, end_missing, i)] = adj0
          P1[(start_missing, end_missing, i)] = P1_0
          P2[(start_missing, end_missing, i)] = P2_0
          A_nn[(start_missing, end_missing, i)] = A_nn0
          F[(start_missing, end_missing, i)]    = F_0
  return(features, adj, routes, P1, P2, A_nn, F)


def construct_adj_fieldsize(A, W, field_size, num_layers):
  res = {}
  for i in range(num_layers):
    t1 = nlg.matrix_power(W, field_size[i])
    t2 = nlg.matrix_power(A, field_size[i])
    t = np.concatenate((t1[:,:,None],t2[:,:,None]), axis=2)
    res[i] = t

  return res



def feature_from_assignment(routing, assignment, num_nodes):
  route = []
  features = np.zeros([num_nodes,6,0])

  index = routing.Start(0)

  cur_node = index % num_nodes
  start_node = cur_node
  goal_node = cur_node
  visited = np.zeros(num_nodes)
  visited[start_node] = 1
  goal = np.zeros([num_nodes, 3])
  goal[goal_node, 0] = 1
  goal[goal_node, 1] = 1
  goal[:, 2] = 1
  state0 = np.zeros([num_nodes, 3])
  state0[cur_node, 0] = 1
  state0[goal_node, 1]  = 1
  visited[cur_node] = 1
  state0[:, 2] = visited
  feature0 = np.c_[state0, goal][:,:,None]
  features = np.concatenate((features,feature0), axis=2)

  while np.sum(visited)!=num_nodes:
    index = assignment.Value(routing.NextVar(cur_node))
    cur_node = index % num_nodes
    route.append(cur_node)
    state0 = np.zeros([num_nodes, 3])
    state0[cur_node, 0] = 1
    state0[goal_node, 1]  = 1
    visited[cur_node] = 1
    state0[:, 2] = visited
    feature0 = np.c_[state0, goal][:,:,None]
    features = np.concatenate((features,feature0), axis=2)
  route.append(start_node)
  return(features, route)


def feature_from_route(route, num_nodes):
  # this doesn't work for some unknown reason...w
  features = np.zeros([num_nodes,6,0])
  cur_node = route[0]
  i = 1
  start_node = cur_node
  goal_node = cur_node
  visited = np.zeros(num_nodes)
  visited[start_node] = 1
  goal = np.zeros([num_nodes, 3])
  goal[goal_node, 0] = 1
  goal[goal_node, 1] = 1
  goal[:, 2] = 1
  state0 = np.zeros([num_nodes, 3])
  state0[cur_node, 0] = 1
  state0[goal_node, 1]  = 1
  visited[cur_node] = 1
  state0[:, 2] = visited
  feature0 = np.c_[state0, goal][:,:,None]
  features = np.concatenate((features,feature0), axis=2)

  while np.sum(visited)!=num_nodes:
    cur_node = route[i]
    state0 = np.zeros([num_nodes, 3])
    state0[cur_node, 0] = 1
    state0[goal_node, 1]  = 1
    visited[cur_node] = 1
    state0[:, 2] = visited
    feature0 = np.c_[state0, goal][:,:,None]
    features = np.concatenate((features,feature0), axis=2)
    i += 1
  route = list(route)
  route.append(start_node)
  route = route[1:]
  return(features, route)




def np_dic_to_json2(dic):
  keys1 = dic.keys()
  new_dic = {}

  for key1 in keys1:
    new_key1 = str(key1[0]) + ',' + str(key1[1])
    new_dic[new_key1] = {}
    keys2 = dic[key1].keys()
    for key2 in keys2:
      new_key2 = str(key2)
      arr = dic[key1][key2]
      if type(arr) == np.ndarray:
        new_dic[new_key1][new_key2] = arr.tolist()
      else:
        new_dic[new_key1][new_key2] = arr
  return(new_dic)
def np_dic_to_json3(dic):
  keys1 = dic.keys()
  new_dic = {}

  for key1 in keys1:
    new_key1 = str(key1[0]) + ',' + str(key1[1])
    new_dic[new_key1] = {}
    keys2 = dic[key1].keys()
    for key2 in keys2:
      new_key2 = str(key2)
      new_dic[new_key1][new_key2] = {}
      keys3 = dic[key1][key2].keys()
      for key3 in keys3:
        new_key3 = str(key3)
        arr = dic[key1][key2][key3]
        if type(arr) == np.ndarray:
          new_dic[new_key1][new_key2][new_key3] = arr.tolist()
        else:
          new_dic[new_key1][new_key2][new_key3] = arr
  return(new_dic)
def json_dic_to_np2(dic):
  keys1 = dic.keys()
  new_dic = {}

  for key1 in keys1:
    new_key1 = tuple(np.array(key1.split(',')).astype(int))
    new_dic[new_key1] = {}
    keys2 = dic[key1].keys()
    for key2 in keys2:
      new_key2 = int(key2)
      arr = dic[key1][key2]
      if type(arr) == list:
        new_dic[new_key1][new_key2] = np.array(arr)
      else:
        new_dic[new_key1][new_key2] = arr
  return(new_dic)
def json_dic_to_np3(dic):
  keys1 = dic.keys()
  new_dic = {}

  for key1 in keys1:
    new_key1 = tuple(np.array(key1.split(',')).astype(int))
    new_dic[new_key1] = {}
    keys2 = dic[key1].keys()
    for key2 in keys2:
      new_key2 = int(key2)
      new_dic[new_key1][new_key2] = {}
      keys3 = dic[key1][key2].keys()
      for key3 in keys3:
        new_key3 = int(key3)
        arr = dic[key1][key2][key3]
        if type(arr) == list:
          new_dic[new_key1][new_key2][new_key3] = np.array(arr)
        else:
          new_dic[new_key1][new_key2][new_key3] = arr

  return(new_dic)









