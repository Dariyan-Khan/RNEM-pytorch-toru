import numpy as np
import torch
import torch.nn as nn


class VanillaRNN(nn.Module):
	
	def __init__(self, batch_size, k, input_size, hidden_size, num_layers=1, device='cpu'):
		super(VanillaRNN, self).__init__()

		self.batch_size = batch_size
		self.k = k
		self.input_size = input_size[-1]
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.device = device
		

		self.rnn = nn.RNN(self.input_size, hidden_size, num_layers=num_layers, batch_first=True)
		self.fc_1 = nn.Linear(hidden_size, self.input_size)
	
	def init_hidden(self):
		# variable of size [num_layers*num_directions, b_sz, hidden_sz]
		if self.device.type == 'cpu':
			# return torch.autograd.Variable(torch.zeros(self.batch_size * self.k, self.hidden_size))
			return torch.autograd.Variable(torch.zeros(self.num_layers, self.batch_size*self.k, self.hidden_size))
		else:
			# return torch.autograd.Variable(torch.zeros(self.batch_size * self.k, self.hidden_size)).cuda()
			return torch.autograd.Variable(torch.zeros(self.num_layers, self.batch_size * self.k, self.hidden_size)).cuda()
	
	def forward(self, x, state):
		output, out_state = self.rnn(x, state) # (input, hidden, and internal state)
		out = self.fc_1(output)

		return out, out_state
