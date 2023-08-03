#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch.linalg import vector_norm

from model import VanillaRNN

def mvn_pdf(x, mu, sigma):
		"""Multivariate pdf under the homoskedastic case"""
		sig_size = x.shape[-1]
		sf = (2*torch.pi)**(-sig_size / 2) * (sigma**sig_size)
		delta_mu = x - mu
		l2_delta_mu = vector_norm(delta_mu, dim=-1, keepdim=True) ** 2
		exp_inner = -0.5 * l2_delta_mu / sigma
		return sf * torch.exp(exp_inner)


class NEM(nn.Module):
	def __init__(self, batch_size, k, input_size, hidden_size, device='cpu'):
		super(NEM, self).__init__()
		self.device = device

		self.inner_rnn = VanillaRNN(batch_size=batch_size, k=k, input_size=input_size, hidden_size=hidden_size, device=device).to(device)

		self.batch_size = batch_size
		self.k = k

		# shape of hidden variables
		gamma_size = input_size[:-1] + (1,)
		self.hidden_size = hidden_size  # 250
		self.input_size = input_size  # (W, H, C)
		self.gamma_size = gamma_size  # (W, H, 1)

		self.pred_init = 0.0

	def init_state(self, dtype=torch.float32):
		"""
		Return a randomly initialized hidden state tuple (h, pred, gamma)

		:return:
			h (B*K, hidden_size)
			pred (B, K, W, H, C)
			gamma (B, K, W, H, 1)
		"""
		device = self.device

		batch_size, K = self.batch_size, self.k

		# initialize h, the latent representation of each object
		h = self.inner_rnn.init_hidden() * self.pred_init

		# initialize pred, the (predicted) true assignment of pixels to objects
		pred = torch.ones(batch_size, K, *self.input_size, dtype=dtype) * self.pred_init

		# initialize gamma, a weight given to pred, with Gaussian distribution
		gamma_shape = [batch_size, K] + list(self.gamma_size)
		gamma = np.absolute(np.random.normal(size=gamma_shape))
		gamma = torch.from_numpy(gamma.astype(np.float32))
		gamma /= torch.sum(gamma, dim=1, keepdim=True)

		# init with all 1 if K = 1
		if K == 1:
			gamma = torch.ones_like(gamma)

		h, pred, gamma = h.to(device), pred.to(device), gamma.to(device)
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
		
		reshaped_masked_deltas = masked_deltas.view(batch_size * K, 1,  M) # Masked deltas get collapsed here, and then when passed hrough encoder they change shape
		preds, h_new = self.inner_rnn.forward(reshaped_masked_deltas, h_old)

		return preds.view(d_size), h_new
	

	def compute_berrnoulli_probabilities(self, predictions, data, sigma=1, epsilon=1e-6):
		"""
		Compute pixelwise loss of predictions (wrt. the data).

		:param predictions: (B, K, W, H, C)
		:param data: (B, 1, W, H, C)
		:return: local loss (B, K, W, H, 1)
		"""
		loss = data * predictions + (1 - data) * (1 - predictions)


		# sum loss over channels
		loss = torch.sum(loss, 4, keepdim=True)

		if epsilon > 0:
			loss += epsilon
		return loss
	

	def compute_em_probabilities(self, predictions, data, sigma = 1, epsilon=1e-6):
		"""
		Compute pixelwise loss of predictions (wrt. the data).

		:param predictions: (B, K, W, H, C)
		:param data: (B, 1, W, H, C)
		:return: local loss (B, K, W, H, 1)
		"""
		
		loss = mvn_pdf(data, predictions, sigma)

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
