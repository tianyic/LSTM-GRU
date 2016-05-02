import numpy as np

from LSTM import LSTM, LSTMblock

"""

Author : Tianyi Chen

Date   : May 2, 2016

Email  : tchen59@jhu.edu
 
"""

def main():
	mem_size = 4
	x_dim = 4
	lstm_net = LSTM( mem_size, x_dim )

	y_list = [ -0.5, 0.2, 0.1, 0.5 ]
	input_val_arr = [ np.random.random(x_dim) for _ in y_list ]

	for cur_iter in xrange(10):

		print 'Current iteration:', cur_iter
		for idx in xrange( len(y_list) ):
		#for idx in xrange(1):
			lstm_net.addx( input_val_arr[idx] )
			# print 'input x:', input_val_arr[idx]

		print 'Calculate gradient terms'
		lstm_net.backwardpropagation( input_val_arr, input_val_arr )
		
		print 'Update parameters'
		lstm_net.updateParams()

		print 'Predict'
		lstm_net.predict( input_val_arr )
		
		print 'Calculate loss'
		print 'Loss:', np.linalg.norm(lstm_net.y - input_val_arr )
		lstm_net.reset()

		


if __name__ == '__main__':
	main()