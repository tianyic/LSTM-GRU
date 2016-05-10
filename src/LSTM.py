#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import math

class LSTM(object):

	"""docstring for LSTM"""
	def __init__(self, mem_dim, x_dim ):
		super(LSTM, self).__init__()

		self.x_dim = x_dim
		self.mem_dim = mem_dim
		self.cec_dim = x_dim + mem_dim
	
		# Weights matrixes for memory cell
		self.W_g = np.random.rand( mem_dim, mem_dim ) * 0.2 + 0.1
		self.W_i = np.random.rand( mem_dim, mem_dim ) * 0.2 + 0.1
		self.W_f = np.random.rand( mem_dim, mem_dim ) * 0.2 + 0.1
		self.W_o = np.random.rand( mem_dim, mem_dim ) * 0.2 + 0.1

		# Weights matrixes for input x
		self.U_g = np.random.rand( mem_dim, x_dim ) * 0.2 + 0.1
		self.U_i = np.random.rand( mem_dim, x_dim ) * 0.2 + 0.1
		self.U_f = np.random.rand( mem_dim, x_dim ) * 0.2 + 0.1
		self.U_o = np.random.rand( mem_dim, x_dim ) * 0.2 + 0.1

		self.V = np.random.rand( x_dim, mem_dim ) * 0.2 + 0.1

		# Biases
		self.b_g = np.zeros( mem_dim )
		self.b_i = np.zeros( mem_dim )
		self.b_f = np.zeros( mem_dim )
		self.b_o = np.zeros( mem_dim )
		
		# Grads
		self.grad_V   = np.zeros_like( self.V )
		self.grad_W_g = np.zeros_like( self.W_g )
		self.grad_W_i = np.zeros_like( self.W_i )
		self.grad_W_f = np.zeros_like( self.W_f )
		self.grad_W_o = np.zeros_like( self.W_o )
		self.grad_U_g = np.zeros_like( self.U_g )
		self.grad_U_i = np.zeros_like( self.U_i )
		self.grad_U_f = np.zeros_like( self.U_f )
		self.grad_U_o = np.zeros_like( self.U_o )
		self.grad_b_g = np.zeros_like( self.b_g )
		self.grad_b_i = np.zeros_like( self.b_i )
		self.grad_b_f = np.zeros_like( self.b_f )
		self.grad_b_o = np.zeros_like( self.b_o )
		self.grad_s   = np.zeros( mem_dim )
		self.grad_c   = np.zeros( mem_dim )

		self.block_list = []
		self.x_list = []
		

	def addx( self, x ):
		
		self.x_list.append(x)
		self.block_list.append( LSTMblock( self, x ) )

		# Get index of most recent x input
		idx = len( self.x_list ) - 1

		if idx == 0:
			self.block_list[idx].forward_propogation()
		else:
			s_old = self.block_list[idx - 1].s
			c_old = self.block_list[idx - 1].c
			self.block_list[idx].forward_propogation( s_old, c_old )
		


	def backwardpropagation( self, x, y_list, Loss = None ):

		T = len( y_list )

		dc_next = np.zeros( self.mem_dim )

		for t in np.arange(T)[::-1]:

			block = self.block_list[t]
			delta_y = block.y - y_list[t]

			self.grad_V = np.outer( delta_y, block.s.T )
			block.grad_s = np.dot( self.V.T, delta_y )

			if t != T - 1:
				nextblock = self.block_list[ t + 1 ]
				dc_next = nextblock.grad_c * nextblock.f
				ds = np.dot( self.W_i.T, tmp_i )
				ds += np.dot( self.W_f.T, tmp_f )
				ds += np.dot( self.W_o.T, tmp_o )
				ds += np.dot( self.W_g.T, tmp_g )
				block.grad_s += ds

			block.grad_c = block.o * block.grad_s + dc_next
			block.grad_o = block.grad_s * block.c
			block.grad_i = block.grad_c * block.g
			block.grad_f = block.grad_c * block.c_old
			block.grad_g = block.i * block.grad_s

			tmp_i = block.grad_i * block.i * ( 1.0 - block.i ) 
			tmp_f = block.grad_f * block.f * ( 1.0 - block.f )
			tmp_o = block.grad_o * block.o * ( 1.0 - block.o )
			tmp_g = block.grad_g * ( 1.0 - block.g ** 2 )			
			
			self.grad_U_i += np.outer( tmp_i, x[t] )
			self.grad_U_f += np.outer( tmp_f, x[t] )
			self.grad_U_g += np.outer( tmp_g, x[t] )
			self.grad_U_o += np.outer( tmp_o, x[t] )

			self.grad_W_i += np.outer( tmp_i, block.s_old )
			self.grad_W_f += np.outer( tmp_f, block.s_old )
			self.grad_W_g += np.outer( tmp_g, block.s_old )
			self.grad_W_o += np.outer( tmp_o, block.s_old )

	def updateParams( self, lr = 0.01 ):

		self.W_i -= lr * self.grad_W_i
		self.W_o -= lr * self.grad_W_o
		self.W_f -= lr * self.grad_W_f
		self.W_g -= lr * self.grad_W_g

		self.U_i -= lr * self.grad_U_i
		self.U_o -= lr * self.grad_U_o
		self.U_f -= lr * self.grad_U_f
		self.U_g -= lr * self.grad_U_g

	def reset( self ):

		self.grad_W_i = np.zeros_like( self.W_i )
		self.grad_W_o = np.zeros_like( self.W_o )
		self.grad_W_f = np.zeros_like( self.W_f )
		self.grad_W_g = np.zeros_like( self.W_g )

		self.grad_U_i = np.zeros_like( self.U_i )
		self.grad_U_o = np.zeros_like( self.U_o )
		self.grad_U_f = np.zeros_like( self.U_f )
		self.grad_U_g = np.zeros_like( self.U_g )	

		self.x_list = []
		self.block_list = []

	def calculateLoss( self, x, label ):
		
		self.y = np.zeros_like( x )
		haty = []
		for i in xrange( len(self.block_list) ):
			block = self.block_list[i]
			block.x = x[i]
			if i == 0:
				block.forward_propogation()
			else:
				nextblock = self.block_list[ i - 1 ]
				s_old = self.block_list[ i - 1 ].s
				c_old = self.block_list[ i - 1 ].c
				block.forward_propogation( s_old, c_old )
			self.y[i] = block.y
			haty.append( block.y[ np.where( label[i] == 1.0 ) ] )
		Loss = 0.0
		Loss += -1 * np.sum( np.log( haty ) )
		
		return Loss

class LSTMblock(object):

	"""docstring for LSTMnode"""
	def __init__( self, lstmnet, x ):
		super(LSTMblock, self).__init__()
		self.lstmnet= lstmnet
		self.s = np.zeros( lstmnet.mem_dim )
		self.g = np.zeros( lstmnet.mem_dim )
		self.c = np.zeros( lstmnet.mem_dim )
		self.i = np.zeros( lstmnet.mem_dim )
		self.f = np.zeros( lstmnet.mem_dim )
		self.x = x
		self.grad_s = np.zeros( lstmnet.mem_dim )
		self.grad_c = np.zeros( lstmnet.mem_dim )
		self.grad_o = np.zeros( lstmnet.mem_dim )
		self.grad_i = np.zeros( lstmnet.mem_dim )
		self.grad_f = np.zeros( lstmnet.mem_dim )


	# forward propogation for bottom data 
	def forward_propogation( self, s_old = None, c_old = None):
		
		if s_old == None:
			s_old = np.zeros_like( self.s )
		if c_old == None:
			c_old = np.zeros_like( self.c )

		self.s_old = s_old
		self.c_old = c_old

		self.i = sigmoid( np.dot( self.lstmnet.U_i, self.x ) + np.dot( self.lstmnet.W_i, s_old ) + self.lstmnet.b_i )
		self.o = sigmoid( np.dot( self.lstmnet.U_o, self.x ) + np.dot( self.lstmnet.W_o, s_old ) + self.lstmnet.b_o )
		self.f = sigmoid( np.dot( self.lstmnet.U_f, self.x ) + np.dot( self.lstmnet.W_f, s_old ) + self.lstmnet.b_f )
		self.g = np.tanh( np.dot( self.lstmnet.U_g, self.x ) + np.dot( self.lstmnet.W_g, s_old ) + self.lstmnet.b_g )
		self.c = self.c_old * self.f + self.g * self.i
		self.s = self.c * self.o
		self.y = softmax( np.dot( self.lstmnet.V, self.s ) )

	
def sigmoid(x): 
    return 1. / (1 + np.exp(-x))

def softmax( x, tau = 1.0 ):
	e = np.exp( np.array(x) / tau )
	return e / np.sum( e )