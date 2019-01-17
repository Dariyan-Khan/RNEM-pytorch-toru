import numpy as np
import utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist

class NEM(nn.Module):
	def __init__(self, inner_rnn, input_size, hidden_size, output_size):
		super(NEM, self).__init__()
		self.inner_rnn = inner_rnn # InnerRNN defined in model.py

		gamma_size = input_size[:-1] + (1,)
		# rnn_input_size = (hidden_size, input_size, gamma_size)
		# rnn_hidden_size = (output_size, input_size, gamma_size)
		# self.rnn = nn.RNN(rnn_input_size, rnn_hidden_size)

		self.hidden_size = hidden_size
		self.input_size = input_size
		self.gamma_size = gamma_size

	def init_state(self, batch_size, K,  dtype=torch.float32):
		h = torch.zeros(batch_size*K, self.hidden_size, dtype=torch.float32)

		pred = torch.zeros(batch_size, K, *self.input_size, dtype=torch.float32)

		# initialize with Gaussian distribution
		gamma_shape = [batch_size, K] + list(self.gamma_size)
		gamma = np.absolute(np.random.normal(size=gamma_shape))
		gamma = torch.from_numpy(gamma.astype(np.float32))
		gamma = torch.sum(gamma, dim=1, keepdim=True)

		# init with all 1 if K = 1
		if K == 1:
			gamma = torch.ones_like(gamma)

		return h, pred, gamma

	@staticmethod
	def delta_predictions(predictions, data):
		"""
		Compute the derivative of the prediction wrt. to the loss.
		For binary and real with just μ this reduces to (predictions - data).

		:param predictions: (B, K, W, H, C)
		Note: This is a list to later support getting both μ and σ.
		:param data: (B, 1, W, H, C)

		:return: deltas (B, K, W, H, C)
		"""
		return data - predictions

	@staticmethod
	def mask_rnn_inputs(rnn_inputs, gamma):
		"""
		Mask the deltas (inputs to RNN) by gamma.
		:param rnn_inputs: (B, K, W, H, C)
		    Note: This is a list to later support multiple inputs
		:param gamma: (B, K, W, H, 1)

		:return: masked deltas (B, K, W, H, C)
		"""
		with torch.no_grad():
			return rnn_inputs * gamma

	def run_inner_rnn(self, masked_deltas, h_old):
		d_size = masked_deltas.size()
		batch_size = d_size[0]
		K = d_size[1]
		M = torch.tensor(self.input_size).prod()
		reshaped_masked_deltas = masked_deltas.view(batch_size * K, M)

		preds, h_new = self.inner_rnn(reshaped_masked_deltas, h_old)

		return preds.view(d_size), h_new

	def compute_em_probabilities(self, predictions, data, epsilon=1e-6):
		"""
		Compute pixelwise loss of predictions (wrt. the data).

		:param predictions: (B, K, W, H, C)
		:param data: (B, 1, W, H, C)
		:return: local loss (B, K, W, H, 1)
		"""
		loss = data * predictions + (1 - data) * (1 - predictions)
		if epsilon > 0:
			loss += epsilon
		return loss

	def e_step(self, predictions, targets):
		probs = self.compute_em_probabilities(predictions, targets)

		# compute the new gamma (E-step)
		gamma = probs / probs.sum(dim=1, keepdim=True)

		return gamma

	def forward(self, x, state):
		# unpack values
		input_data, target_data = x
		h_old, preds_old, gamma_old = state

		# compute differences between prediction and input
		deltas = self.delta_predictions(preds_old, input_data)

		# mask with gamma
		masked_deltas = self.mask_rnn_inputs(deltas, gamma_old)

		# compute new predictions
		preds, h_new = self.run_inner_rnn(masked_deltas, h_old)

		# compute the new gammas
		gamma = self.e_step(preds, target_data)

		# pack and return
		outputs = (h_new, preds, gamma)

		return outputs, outputs