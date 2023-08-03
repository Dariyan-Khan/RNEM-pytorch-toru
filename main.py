import argparse
import os
import time

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader, Subset
from torch.linalg import vector_norm
from torch.distributions.multivariate_normal import MultivariateNormal




import utils
from data import Data, collate
from nem import NEM
from utils import MSELoss, KLDivLossNormal

# Device configuration
use_gpu = None
device = None

args = None


### helper functions


def mvn_squared_error_loss(mu, x, sigma=1.0, include_constant=False):
	"""Loss function for multivatiate gaussian in the homoskedastic case"""

	sig_size =  x.shape[-1]
	det_sig =  torch.tensor(sigma**sig_size) #torch.linalg.det(sigma)

	#inv_sig = torch.clamp(torch.linalg.inv(sigma), 1e-6, 1e6)
	mean_delta = x - mu
	mean_delta = torch.squeeze(mean_delta)
	l2_mu_diff = vector_norm(mean_delta, dim=-1)**2
	#mean_delta_t = torch.t(mean_delta)
	# res = torch.log(torch.clamp(det_sig, 1e-6, 1e6)) + torch.matmul(torch.matmul(mean_delta_t, inv_sig), mean_delta)
	res = (include_constant * -sig_size * torch.log(torch.tensor(2 * torch.pi))) + torch.log(torch.clamp(det_sig, 1e-6, 1e6)) + l2_mu_diff / sigma# (torch.matmul(mean_delta_t, mean_delta) / sigma)

	if use_gpu:
		return res.cuda()
	else:
		return res


def kl_loss_mvn(mu_1, mu_2, sigma_1=1.0, sigma_2=1.0):
	sig_size = mu_1.shape[-1]
	det_sig_1 = torch.tensor(sigma_1**sig_size) # torch.linalg.det(sigma_1)
	det_sig_2 = torch.tensor(sigma_2**sig_size) # torch.linalg.det(sigma_2)
	# inv_sig_2 = torch.clamp(torch.linalg.inv(sigma_2), 1e-6, 1e6)
	# d = sigma_2.shape[-1]
	mu_diff = mu_1 - mu_2
	mu_diff = torch.squeeze(mu_diff)
	l2_mu_diff = vector_norm(mu_diff, dim=-1)**2

	# mu_diff_t = torch.t(mu_diff)
	# res = 0.5 * (torch.log(det_sig_1 / det_sig_2) + torch.trace(torch.matmul(inv_sig_2, sigma_1)) + torch.matmul(torch.matmul(mu_diff_t, inv_sig_2), mu_diff) - d)
	# res = -(torch.log(det_sig_1 / det_sig_2) + (sigma_1*sig_size) / sigma_2 + (torch.matmul(mu_diff_t, mu_diff) / sigma_2) - sig_size)
	res = -(torch.log(det_sig_1 / det_sig_2) + (sigma_1*sig_size) / sigma_2 + (l2_mu_diff / sigma_2) - sig_size)

	if use_gpu:
		return res.cuda()
	else:
		return res


def add_mvn_noise(data, noise_type="mvn", sigma=0.25, noise_prob=0.2):
	"""Given our data is multivariatte normal, we add noise similar to 
	how it is done in the RTagger paper by just adding a vector from a 
	MVN(0, sigma*I) distribution, but masked by bernoulli noise to help prevent
	overcapacity"""

	#	TxBxKxWxHxC

	if noise_type == None:
		return data
	
	elif noise_type != "mvn":
		raise NotImplementedError("This noise type is not supported")
	
	else:
		data_shape = data.shape
		
		num_channels = data.shape[-1]

		noise_mean = torch.zeros(num_channels)
		noise_covariance = sigma * torch.eye(num_channels)

		multivariate_normal = MultivariateNormal(noise_mean, noise_covariance)

		mvn_sample = multivariate_normal.sample(sample_shape=data_shape[:-1]).to(device)


		noise_dist = dist.Bernoulli(probs=noise_prob)
		noise_mask = noise_dist.sample(data_shape[:-1] + (1,)).to(device)

		corrupted_data = data + (noise_mask * mvn_sample)

		return corrupted_data


def compute_normal_prior():
	"""
	Compute Normal prior over the input data with p = 0.0
	"""
	# convert to cuda tensor on GPU
	return torch.zeros(1, 1, 1, 1, 1).to(device)


def compute_outer_loss(mu, gamma, target, prior, collision):

	# # use binomial cross entropy as intra loss
	# intra_criterion = BCELoss().to(device)

	# # use KL divergence as inter loss
	# inter_criterion = KLDivLoss().to(device)

	# intra_loss = intra_criterion(mu, target, use_gpu=use_gpu)
	# inter_loss = inter_criterion(prior, mu, use_gpu=use_gpu)

	intra_loss = mvn_squared_error_loss(mu, target)
	inter_loss = kl_loss_mvn(prior, mu)

	batch_size = target.size()[0]

	# compute rel losses
	r_intra_loss = torch.div(torch.sum(collision * intra_loss * gamma.detach()), batch_size)
	r_inter_loss = torch.div(torch.sum(collision * inter_loss * (1. - gamma.detach())), batch_size)

	# compute normal losses
	intra_loss = torch.div(torch.sum(intra_loss * gamma.detach()), batch_size)
	inter_loss = torch.div(torch.sum(inter_loss * (1.0 - gamma.detach())), batch_size)

	total_loss = intra_loss + inter_loss
	r_total_loss = r_intra_loss + r_inter_loss

	return total_loss, intra_loss, inter_loss, r_total_loss, r_intra_loss, r_inter_loss


def compute_outer_ub_loss(pred, target, prior, collision):
	max_pred, _ = torch.max(pred, dim=1)
	max_pred = torch.unsqueeze(max_pred, 1)

	# use binomial cross entropy as intra loss
	intra_criterion =  MSELoss().to(device) #BCELoss().to(device)

	# use KL divergence as inter loss
	inter_criterion = KLDivLossNormal().to(device) # KLDivLoss().to(device)

	intra_ub_loss = intra_criterion(max_pred, target)
	inter_ub_loss = inter_criterion(prior, max_pred)

	batch_size = target.size()[0]

	r_intra_ub_loss = torch.div(torch.sum(collision * intra_ub_loss), batch_size)
	r_inter_ub_loss = torch.div(torch.sum(collision * inter_ub_loss), batch_size)

	intra_ub_loss = torch.div(torch.sum(intra_ub_loss), batch_size)
	inter_ub_loss = torch.div(torch.sum(inter_ub_loss), batch_size)

	total_ub_loss = intra_ub_loss + inter_ub_loss
	r_total_ub_loss = r_intra_ub_loss + r_inter_ub_loss

	return total_ub_loss, intra_ub_loss, inter_ub_loss, r_total_ub_loss, r_intra_ub_loss, r_inter_ub_loss


### run epoch iterations


def nem_iterations(input_data, target_data, nem_model, optimizer, collisions=None, train=True):
	# compute Bernoulli prior of pixels
	prior = compute_normal_prior()

	# output
	hidden_state = nem_model.init_state(dtype=torch.float32)
	# hidden_state = (nem_model.h, nem_model.pred, nem_model.gamma)
	outputs = [hidden_state]

	# record losses
	total_losses = []
	total_ub_losses = []
	r_total_losses = []
	r_total_ub_losses = []
	other_losses = []
	other_ub_losses = []
	r_other_losses = []
	r_other_ub_losses = []

	loss_step_weights = [1.0] * args.nr_steps

	for t, loss_weight in enumerate(loss_step_weights):
		# model should predict the next frame
		inputs = (input_data[t], target_data[t + 1])



		assert len(input_data[t]) == len(target_data[t + 1]), \
			"Input data and target data must have the same shape"

		# print("inputs", inputs, inputs[0].size(), inputs[1].size())

		# forward pass
		hidden_state, output = nem_model.forward(inputs, hidden_state)
		theta, pred, gamma = output

		# use collision data
		collision = torch.zeros(1, 1, 1, 1, 1).to(device) if collisions is None else collisions[t]

		# compute NEM losses
		total_loss, intra_loss, inter_loss, r_total_loss, r_intra_loss, r_inter_loss \
			= compute_outer_loss(pred, gamma, target_data[t + 1], prior, collision=collision)

		# compute estimated loss upper bound (which doesn't use E-step)
		total_ub_loss, intra_ub_loss, inter_ub_loss, r_total_ub_loss, r_intra_ub_loss, r_inter_ub_loss \
			= compute_outer_ub_loss(pred, target_data[t + 1], prior, collision=collision) #where total ub loss is 

		total_losses.append(loss_weight * total_loss)
		total_ub_losses.append(loss_weight * total_ub_loss)

		r_total_losses.append(loss_weight * r_total_loss)
		r_total_ub_losses.append(loss_weight * r_total_ub_loss)

		other_losses.append(torch.stack((total_loss, intra_loss, inter_loss)))
		other_ub_losses.append(torch.stack((total_ub_loss, intra_ub_loss, inter_ub_loss)))

		r_other_losses.append(torch.stack((r_total_loss, r_intra_loss, r_inter_loss)))
		r_other_ub_losses.append(torch.stack((r_total_ub_loss, r_intra_ub_loss, r_inter_ub_loss)))

		outputs.append(output)  # thetas, preds, gammas

		# delete used variables to save memory space
		del theta, pred, gamma
		del intra_loss, inter_loss, r_intra_loss, r_inter_loss
		del r_intra_ub_loss, r_inter_ub_loss, intra_ub_loss, inter_ub_loss

		if t % args.step_log_per_iter == 0:
			print("Step [{}/{}], Loss: {:.4f}".format(t, args.nr_steps, total_loss))

	# collect outputs
	thetas, preds, gammas = zip(*outputs) # gammas is of shape TxBxKxWxHx1
	thetas = torch.stack(thetas)
	preds = torch.stack(preds)
	gammas = torch.stack(gammas)


	# # collect outputs for graph drawing
	# outputs = {
	# 	'inputs': target_data.cpu(),
	# 	'corrupted': input_data.cpu(),
	# 	'gammas': gammas.cpu(),
	# 	'preds': preds.cpu(),
	# }
	#
	# idx = [0, 1, 2]  # sample ids to generate plots
	# create_rollout_plots('training', outputs, idx)

	other_losses = torch.stack(other_losses)
	other_ub_losses = torch.stack(other_ub_losses)
	r_other_losses = torch.stack(r_other_losses)
	r_other_ub_losses = torch.stack(r_other_ub_losses)

	total_loss = torch.sum(torch.stack(total_losses)) / np.sum(loss_step_weights)
	total_ub_loss = torch.sum(torch.stack(total_ub_losses)) / np.sum(loss_step_weights)
	r_total_loss = torch.sum(torch.stack(r_total_losses)) / np.sum(loss_step_weights)
	r_total_ub_loss = torch.sum(torch.stack(r_total_ub_losses)) / np.sum(loss_step_weights)

	if train:
		# backward pass and optimize
		optimizer.zero_grad()
		total_loss.backward()
		optimizer.step()

	return total_loss, total_ub_loss, r_total_loss, r_total_ub_loss, thetas, preds, gammas, other_losses, \
		   other_ub_losses, r_other_losses, r_other_ub_losses


def run_epoch(epoch, nem_model, optimizer, dataloader, train=True):
	# adjust mode
	if train:
		nem_model.train()
	else:
		nem_model.eval()

	losses = []
	ub_losses = []
	r_losses = []
	r_ub_losses = []
	others = []
	others_ub = []
	r_others = []
	r_others_ub = []
	gamma_list = []
	if train:
		# run through all data batches
		for i, data in enumerate(dataloader):

			print(f"data: {data[0].keys()}")

			# assert False

			if args.subset > 0 and  i == args.subset:
				break
			# per batch
			features = data[0]['features']
			groups = data[0]['groups'] if 'groups' in data[0] else None
			collisions = data[0]['collision'] if 'collisions' in data[0] else None
			
			features = features.to(device)   # (T, B, K, W, H, C)
			if groups is not None:
				groups = groups.to(device)
			if collisions is not None:
				collisions = collisions.to(device)

			features_corrupted = add_mvn_noise(features, noise_type=None)   # (T, B, K, W, H, C)

			# show_image(features[0].cpu(), 0, 0)
			# show_image(features_corrupted[0].cpu(), 0, 0)

			t1 = time.time()
			out = nem_iterations(features_corrupted, features, nem_model, optimizer)
			t2 = time.time() - t1
			print("time taken for batch", i, "=", t2)

			losses.append(out[0].data.cpu().numpy())
			ub_losses.append(out[1].data.cpu().numpy())

			# total relational losses (and upperbound)
			r_losses.append(out[2].data.cpu().numpy())
			r_ub_losses.append(out[3].data.cpu().numpy())

			# other losses (and upperbound)
			others.append(out[4].data.cpu().numpy())
			others_ub.append(out[5].data.cpu().numpy())

			# other relational losses (and upperbound)
			r_others.append(out[6].data.cpu().numpy())
			r_others_ub.append(out[7].data.cpu().numpy())

			gamma_means = torch.mean(out[6], dim=0, keepdim=False)
			gamma_list.append(gamma_means)

			if epoch % args.log_per_iter == 0 and i % args.log_per_batch == 0:
				print("Epoch [{}] Batch [{}], Loss: {:.4f}".format(epoch, i, losses[-1]))
				torch.save(nem_model.state_dict(),
						   os.path.abspath(os.path.join(args.save_dir, 'epoch_{}_batch_{}.pth'.format(epoch, i))))

	else:
		# disable autograd if eval
		with torch.no_grad():
			for i, data in enumerate(dataloader):

				if args.subset > 0 and  i == args.subset:
					break

				# per batch
				# per batch
				features = data[0]['features']
				groups = data[0]['groups'] if 'groups' in data[0] else None
				collisions = data[0]['collision'] if 'collisions' in data[0] else None

				features = features.to(device)
				if groups is not None:
					groups = groups.to(device)
				if collisions is not None:
					collisions = collisions.to(device)

				features_corrupted = add_mvn_noise(features, noise_type=None)

				t1 = time.time()
				out = nem_iterations(features_corrupted, features, nem_model, optimizer, train=False)
				t2 = time.time() - t1
				print("time taken for epoch", i, "=", t2)

				losses.append(out[0].data.cpu().numpy())
				ub_losses.append(out[1].data.cpu().numpy())

				# total relational losses (and upperbound)
				r_losses.append(out[2].data.cpu().numpy())
				r_ub_losses.append(out[3].data.cpu().numpy())

				# other losses (and upperbound)
				others.append(out[4].data.cpu().numpy())
				others_ub.append(out[5].data.cpu().numpy())

				# other relational losses (and upperbound)
				r_others.append(out[6].data.cpu().numpy())
				r_others_ub.append(out[7].data.cpu().numpy())

				gamma_means = torch.mean(out[6], dim=0, keepdim=False)
				gamma_list.append(gamma_means)
	
	gammas = torch.stack(gamma_list, dim=0)


	# build log dict
	log_dict = {
		'loss': float(np.mean(losses)),
		'ub_loss': float(np.mean(ub_losses)),
		'r_loss': float(np.mean(r_losses)),
		'r_ub_loss': float(np.mean(r_ub_losses)),
		'others': np.mean(others, axis=0),
		'others_ub': np.mean(others_ub, axis=0),
		'r_others': np.mean(r_others, axis=0),
		'r_others_ub': np.mean(r_others_ub, axis=0),
		"gammas": gammas
	}

	return log_dict


### log computation results

def log_log_dict(phase, log_dict):
	fp = os.path.abspath(os.path.join(args.log_dir, 'log_{}'.format(phase)))
	with open(fp, "a") as f:
		for k, v in log_dict.items():
			f.write("{}: {}\n".format(k, v))


def print_log_dict(log_dict, s_loss_weights, dt_s_loss_weights):
	dt = args.dt

	loss = log_dict['loss']
	ub_loss = log_dict['ub_loss']
	r_loss = log_dict['r_loss']
	r_ub_loss = log_dict['r_ub_loss']
	other_losses = log_dict['others']
	other_ub_losses = log_dict['others_ub']
	r_other_losses = log_dict['r_others']
	r_other_ub_losses = log_dict['r_others_ub']

	print("Loss: %.3f (UB: %.3f), Relational Loss: %.3f (UB: %.3f)" % (loss, ub_loss, r_loss, r_ub_loss))  # The losses

	try:
		print("    other losses: {}".format(", ".join(["%.2f (UB: %.2f)" %
													   (other_losses[:, i].sum(0) / s_loss_weights,
														other_ub_losses[:, i].sum(0) / s_loss_weights)
													   for i in range(len(other_losses[0]))])))

		print("        last {} steps avg: {}".format(dt, ", ".join(["%.2f (UB: %.2f)" %
																	(other_losses[-dt:, i].sum(0) / dt_s_loss_weights,
																	 other_ub_losses[-dt:, i].sum(0) / dt_s_loss_weights)
																	for i in range(len(other_losses[0]))])))

		print("    other relational losses: {}".format(", ".join(["%.2f (UB: %.2f)" %
																  (r_other_losses[:, i].sum(0) / s_loss_weights,
																   r_other_ub_losses[:, i].sum(0) / s_loss_weights)
																  for i in range(len(r_other_losses[0]))])))

		print("        last {} steps avg: {}".format(dt, ", ".join(["%.2f (UB: %.2f)" %
																	(r_other_losses[-dt:, i].sum(0) / dt_s_loss_weights,
																	 r_other_ub_losses[-dt:, i].sum(0) / dt_s_loss_weights)
																	for i in range(len(r_other_losses[0]))])))
	except:
		pass


### Main functions


def run():
	if use_gpu:
		torch.set_default_tensor_type('torch.cuda.FloatTensor')

	for dir in [args.log_dir, args.save_dir]:
		utils.create_directory(dir)

	# only clear log_dir
	# utils.clear_directory(args.log_dir)

	attribute_list = ('features', 'groups')
	nr_iters = args.nr_steps + 1

	# set up input data
	train_inputs = Data(args.data_name, "training", args.batch_size, nr_iters, attribute_list)
	valid_inputs = Data(args.data_name, "validation", args.batch_size, nr_iters, attribute_list)

	print(f"train shape: {train_inputs[0]['features'].shape}")

	#print(len(train_inputs))

	# if args.subset > 0:
	# 	train_inputs = Subset(train_inputs, list(range(args.subset)))
	# 	valid_inputs = Subset(valid_inputs, list(range(args.subset)))

	train_dataloader = DataLoader(dataset=train_inputs, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
	valid_dataloader = DataLoader(dataset=valid_inputs, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)

	# get dimensions of data
	input_shape = train_inputs.data["features"].shape
	W, H, C = list(input_shape)[-3:]

	# set up model
	train_model = NEM(batch_size=args.batch_size,
					  k=args.k,
					  input_size=(W, H, C),
					  hidden_size=args.inner_hidden_size,
					  device=device)

	if use_gpu:
		train_model = nn.DataParallel(train_model)

	train_model.to(device)

	if args.saved_model != None and args.saved_model != "":
		# load trained NEM model if exists
		saved_model_path = os.path.join(args.save_dir, args.saved_model)
		assert os.path.isfile(saved_model_path), "Path to model does not exist"
		train_model.load_state_dict(torch.load(saved_model_path))

	# set up optimizer
	optimizer = torch.optim.Adam(list(train_model.parameters()) + list(train_model.inner_rnn.parameters()), lr=args.lr)

	# training
	best_valid_loss = np.inf

	# prepare weights for printing out logs
	loss_step_weights = [1.0] * args.nr_steps
	s_loss_weights = np.sum(loss_step_weights)
	dt_s_loss_weights = np.sum(loss_step_weights[-args.dt:])

	for epoch in range(1, args.max_epoch + 1):
		# produce print-out
		print("\n" + 50 * "%" + "    EPOCH {}   ".format(epoch) + 50 * "%")

		# run train epoch
		print("=" * 10, "Train", "=" * 10)
		log_dict = run_epoch(epoch, train_model, optimizer, train_dataloader, train=True)

		log_log_dict('training', log_dict)
		print_log_dict(log_dict, s_loss_weights, dt_s_loss_weights)

		# run eval epoch
		print("=" * 10, "Eval", "=" * 10)
		log_dict = run_epoch(epoch, train_model, optimizer, valid_dataloader, train=False)

		log_log_dict('validation', log_dict)
		print_log_dict(log_dict, s_loss_weights, dt_s_loss_weights)

		if log_dict['loss'] < best_valid_loss:
			best_valid_loss = log_dict['loss']
			print("Best validation loss improved to %.03f" % best_valid_loss)
			print("Best valid epoch [{}/{}]".format(epoch, args.max_epoch + 1))
			torch.save(train_model.state_dict(), os.path.abspath(os.path.join(args.save_dir, 'best.pth')))
			print("=== Saved to:", args.save_dir)

		if epoch % args.log_per_iter == 0:
			print("Epoch [{}/{}], Loss: {:.4f}".format(epoch, args.max_epoch, log_dict['loss']))
			torch.save(train_model.state_dict(),
					   os.path.abspath(os.path.join(args.save_dir, 'epoch_{}.pth'.format(epoch))))

		if np.isnan(log_dict['loss']):
			print("Early Stopping because validation loss is nan")
			break


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_name', type=str, default='balls3curtain64')
	parser.add_argument('--log_dir', type=str, default='./debug')
	parser.add_argument('--save_dir', type=str, default='./trained_model')
	parser.add_argument('--nr_steps', type=int, default=30)
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--max_epoch', type=int, default=500)
	parser.add_argument('--dt', type=int, default=10)
	parser.add_argument('--noise_type', type=str, default='bitflip')
	parser.add_argument('--log_per_iter', type=int, default=1)
	parser.add_argument('--log_per_batch', type=int, default=10)
	parser.add_argument('--step_log_per_iter', type=int, default=10)
	parser.add_argument('--k', type=int, default=5)
	parser.add_argument('--inner_hidden_size', type=int, default=250)
	parser.add_argument('--saved_model', type=str, default='')
	parser.add_argument('--rollout_steps', type=int, default=10)
	parser.add_argument('--subset', type=int, default=0)
	parser.add_argument('--usage', '-u', choices=['train', 'eval', 'rollout'], required=True)

	### for testing purpose
	parser.add_argument('--cpu', default=False, action='store_true')

	args = parser.parse_args()
	print("=== Arguments ===")
	print(args)
	print()

	if args.cpu:
		use_gpu = False
	else:
		use_gpu = torch.cuda.is_available()
	device = torch.device('cuda' if use_gpu else 'cpu')

	if args.usage == 'train':
		run()
	else:
		raise ValueError


# Command I use: time python main.py -u train --max_epoch 2 --nr_steps 2 --subset 2

# python main.py -u train --max_epoch 2 --nr_steps 2 --batch_size 2 --data_name torch_train_obs_10