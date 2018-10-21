# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 17:33:04 2018

@author: Luca Morss
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import socket
import struct

recvport = 50001 #set same in Simulink
sendport = 50002

learn_mode = True
test_mode  = True

filePath_save = r'C:/Users/Morss_Reinforce_CartPole_NN_'
filePath = r'C:/Users/Morss_Reinforce_Cartpole_NN'

## ANN topology
action_size=5
hidden_size =40
state_size = 5
## ANN hyperparameters
alpha = 1e-4

## ANN initilization
bias_init_mean=.5,
bias_init_std=.00
kernel_init_mean = 0.001
kernel_init_std = 0.#2/np.sqrt(state_size)
## RL algorithm parameters
gamma = .9999

## learning setup

total_episodes = 1000000 #Set total number of episodes to train agent on.
max_ep = 351
update_frequency = 1
std_noise = .05


pos_min = -10
pos_max = 10
pos_dot_min = -7
pos_dot_max = 7
phi_min = -0.5
phi_max = 0.5
phi_dot_min = -10
phi_dot_max = 10

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    G = 0
    for t in reversed(range(0, r.size)):
        G = G * gamma + r[t]
        discounted_r[t] = G
    return discounted_r

class actor():
    def __init__(self, alpha, state_size, action_size, hidden_size):
        
        ## Sample action ##
        self.state = tf.placeholder(tf.float32, 
                                    shape=[None,state_size]) # waits for feed_dict
        noise = tf.random_normal(shape=tf.shape(self.state), mean=0.0, stddev=std_noise, dtype=tf.float32)
        hidden1     = tf.layers.dense(self.state+noise,
                                      units=hidden_size,
                                      bias_initializer=tf.random_normal_initializer(bias_init_mean,bias_init_std),
                                      kernel_initializer=tf.random_normal_initializer(kernel_init_mean,kernel_init_std),
                                      activation = tf.nn.elu)
#        hidden2     = tf.layers.dense(hidden1,
#                                      units=hidden_size,
#                                      bias_initializer=None, 
#                                      kernel_initializer=None,#tf.random_normal_initializer(kernel_init_mean,kernel_init_std),                                 
#                                      activation = tf.nn.elu)
        self.output = tf.layers.dense(hidden1,
                                      action_size, 
                                      bias_initializer=tf.random_normal_initializer(bias_init_mean,bias_init_std),#None, 
                                      activation = tf.nn.softmax)
        
        ## Train ##
        self.rewardholder = tf.placeholder(shape=[None], dtype=tf.float32) # waits for feed_dict
        self.actionholder = tf.placeholder(shape=[None], dtype=tf.int32)   # waits for feed_dict
        
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.actionholder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)        
        
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.rewardholder)
        var_list  = tf.trainable_variables()
        
        self.gradient_holders = []
        for idx,var in enumerate(var_list):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        
        self.gradients = tf.gradients(self.loss,var_list)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=alpha)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,var_list))
        #apply_gradients and compute_gradients can be substituted by  minimize()
#        self.train = optimizer.minimize(self.loss, aggregation_method=None, colocate_gradients_with_ops=False)
#        self.grads_and_vars = optimizer.compute_gradients(self.loss, var_list)
#        self.train = optimizer.apply_gradients(self.grads_and_vars) 
        
tf.reset_default_graph()
MC_Actor = actor(alpha= alpha, state_size=state_size, action_size=action_size, hidden_size=hidden_size)
init = tf.global_variables_initializer()

a_send11 = 0.
a_send1 = a_send11

#####################################################
################ SOCKET INITIATION ##################
"""
Receiver Socket
"""
receiversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # STREAM -> TCP/IP
receiversocket.bind(('localhost',recvport))
receiversocket.listen(5)
(recvclientsocket, address) = receiversocket.accept()
"""
Send Socket
"""
sendsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # STREAM -> TCP/IP
sendsocket.bind(('localhost',sendport))
sendsocket.listen(5)
(sendclientsocket, address) = sendsocket.accept()
#print('check 4')
data1 = recvclientsocket.recv(8)
data1 = struct.unpack('>d', data1)
print('x:', data1)
pos = data1[0]
data2 = recvclientsocket.recv(8)
data2 = struct.unpack('>d', data2)
print('x_dot:', data2)
pos_dot = data2[0]
data3 = recvclientsocket.recv(8)
data3 = struct.unpack('>d', data3)
print('phi:', data3)
phi = data3[0]
data4 = recvclientsocket.recv(8)
data4 = struct.unpack('>d', data4)
print('phi_dot:', data4)
phi_dot = data4[0]
msg = struct.pack('>d', a_send1)	# >d stands for bigendian # struct.pack packs float as bytes
sendclientsocket.send(msg)      

################ SOCKET INITIATED ###################
#####################################################

s1 = [pos, pos_dot, phi, phi_dot, a_send1]
s=s1

with tf.Session() as sess:
    if learn_mode == True:
        saver = tf.train.Saver()
    if test_mode == True:
        saver = tf.train.import_meta_graph(r'C:\Users\Chewbacca8\Documents\MA\Code\REINFORCE_dActs_run15_-4649.meta')
#        saver.restore(sess,tf.train.latest_checkpoint(''))
    sess.run(init)
    
    i = 0
    total_return = []
    G_avrg = 10.
    total_length = []
    total_SSE     = []
    best_reward = -np.inf
    saver_step   = 1.
    # Set all gradients to zero
    gradBuffer = sess.run(tf.trainable_variables())
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
    while i < total_episodes:
        G = 0
        
        SSE = 0
        ep_history  = []
        pos_ep      = []
        posdot_ep   = []
        phi_ep      = []
        phidot_ep   = []
        action_ep   = []
        a = 0. # if using tanh as activation of output layer
        a_send1 = 0.
        for j in range(max_ep):
            a_send = a_send1
            a_dist = sess.run(MC_Actor.output, feed_dict={MC_Actor.state:[s]}) # feed actor with states s to produce action a
            a = np.random.choice(a_dist[0],p=a_dist[0])
            a = np.argmax(a_dist == a) #returns indices of the maximum values along an axis
            aa1 = a.astype(float)
            aa1 = (a-np.floor(action_size/2))/np.floor(action_size/2)
            a_send1 += .1*aa1
            np.clip(a_send1, -1, 1)
            data1 = recvclientsocket.recv(8)
            try:
                data1 = struct.unpack('>d', data1)
            except:
                receiversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # STREAM -> TCP/IP
                receiversocket.bind(('localhost',recvport))
                receiversocket.listen(5)
                (recvclientsocket, address) = receiversocket.accept()
                """
                Send Socket
                """
                sendsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # STREAM -> TCP/IP
                sendsocket.bind(('localhost',sendport))
                sendsocket.listen(5)
                (sendclientsocket, address) = sendsocket.accept()
                data1 = recvclientsocket.recv(8)
                data1 = struct.unpack('>d', data1)
            pos = data1[0] 
            data2 = recvclientsocket.recv(8)
            data2 = struct.unpack('>d', data2)
            pos_dot = data2[0]           
            data3 = recvclientsocket.recv(8)
            data3 = struct.unpack('>d', data3)
            phi = data3[0]            
            data4 = recvclientsocket.recv(8)
            data4 = struct.unpack('>d', data4)
            phi_dot = data4[0]
            msg = struct.pack('>d',a_send1)	# >d stands for bigendian # struct.pack packs float as bytes
            sendclientsocket.send(msg) 
                
            s1 = [pos, pos_dot, phi, phi_dot, a_send1] 
            s1[0] = -1 + 2*(s1[0] - pos_min)/(pos_max - pos_min)
            s1[1] = -1 + 2*(s1[1] - pos_dot_min)/(pos_dot_max - pos_dot_min)
            s1[2] = -1 + 2*(s1[2] - phi_min)/(phi_max - phi_min)
            s1[3] = -1 + 2*(s1[3] - phi_dot_min)/(phi_dot_max - phi_dot_min)
            r = (1. - s1[0]**2 - .02*(5-pos)**2) #- s1[3]**2 #- .5*(a_send1-a_send)**2
            if j >= 50:
                r+=5
            if j>= 100:
                r+=5
            square_error = - s1[0]**2
            ep_history.append([s,a,r,s1])
            phi_ep.append(phi)
            phidot_ep.append(phi_dot)
            pos_ep.append(pos)
            posdot_ep.append(pos_dot)
            action_ep.append(a_send1)
            s=s1
            G += r
            
            SSE+= square_error
            #grad= tf.trainable_variables()
            #print(i,j,a_send1,s,r,grad[0])
            
            if abs(pos) > 10 or abs(phi) > 0.5:
                ep_history = np.array(ep_history)
                ep_history[:,2] = discount_rewards(ep_history[:,2])
                feed_dict={MC_Actor.state:np.vstack(ep_history[:,0]),
                           MC_Actor.actionholder:ep_history[:,1],
                           MC_Actor.rewardholder:ep_history[:,2]}
                grads = sess.run(MC_Actor.gradients, feed_dict=feed_dict)
                for idx,grad in enumerate(grads):
                    gradBuffer[idx] += grad
                
                if G > best_reward and learn_mode == True:
                    if saver_step == 1:
                        best_reward = G
                        saver.save(sess, 
                                   filePath_save,
                                   write_meta_graph=True,
                                   global_step=i)

                    else:
                        best_reward = G
                        saver.save(sess, 
                                   filePath_save, 
                                   write_meta_graph=True,
                                   global_step=i)
                    np.savetxt(filePath+str(i)+'best_ep_roll.csv',phi_ep   , fmt='%.5f', delimiter=',')
                    np.savetxt(filePath+str(i)+'best_ep_rollrate.csv',phidot_ep   , fmt='%.5f', delimiter=',')
                    np.savetxt(filePath+str(i)+'best_ep_pos.csv',pos_ep   , fmt='%.5f', delimiter=',')
                    np.savetxt(filePath+str(i)+'best_ep_posdot.csv',posdot_ep   , fmt='%.5f', delimiter=',')
                    np.savetxt(filePath+str(i)+'best_ep_action.csv',action_ep   , fmt='%.5f', delimiter=',')

                    saver_step = i
                    print(i,j/100., G)
                
                if i % update_frequency == 0 and i!=0:
                    feed_dict= dictionary = dict(zip(MC_Actor.gradient_holders, gradBuffer))
                    _=sess.run(MC_Actor.update_batch, feed_dict=feed_dict)
                    for idx,grad in enumerate(grads):
                        gradBuffer[idx] += grad                                              
                total_return.append(G)
                total_length.append(j)
                total_SSE.append(SSE)
                break
        if i % 100 == 0:
            G_avrg=np.mean(total_length[-100:])
            print(np.mean(total_length[-100:]), np.mean(total_return[-100:]))
            np.savetxt(filePath+'SSE.csv',total_SSE   , fmt='%.5f', delimiter=',')
            np.savetxt(filePath+  't.csv',total_length, fmt='%.5f', delimiter=',')
            
        i+=1

recvclientsocket.close()
sendclientsocket.close()