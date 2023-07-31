import torch
import main
import model
import h5py
import pickle
import numpy as np


torch.manual_seed(42)


def reshape(shape, x):
	batch_size = x.size()[0]

	if shape == -1:
		return x.view(batch_size, -1)
	else:
		reshape_size = (batch_size,) + shape
		return x.view(reshape_size)


def test_reshape():
	random_tensor = torch.randn(320, 64, 64, 1)
	a = reshape(-1, random_tensor)
	print(a.shape)
	

def test_mvn_squared_error_loss():
	x, sigma, mu = torch.Tensor([1,2]), torch.Tensor([[1,0], [0,1]]), torch.Tensor([1,3])
	a = main.mvn_squared_error_loss(x, sigma, mu)
	print(a)

# test_mvn_squared_error_loss()

# test_reshape()

def test_kl_loss_mvn():
	mu_1, sigma_1 = torch.Tensor([1,2]), torch.Tensor([[1,0], [0,1]])
	mu_2, sigma_2 = torch.Tensor([1,2]), torch.Tensor([[1,0], [0,1]])

	a = main.kl_loss_mvn(mu_1, sigma_1, mu_2, sigma_2)

	print(a)

def test_new_model():
	batch_size = 32
	k = 3
	input_size = 4
	hidden_size = 2
	irnn = model.InnerRNN(batch_size, k, input_size, hidden_size)
	pred_init = 0
	state = irnn.inner_rnn.init_hidden() * pred_init
	x = torch.randn((32, 4, 5, 6))


	out = irnn(x, state)

def print_h5_file():
	filename = "./data/balls4mass64.h5"
	file = h5py.File(filename, 'r')
	i = 0
	group = file["training"]

	members = ["collisions", "events", "features", "groups", "positions", "velocities"]

	print(group["features"][0])

def slice_test():
	sequence_length = 2
	batch_size = 3
	idx = 1

	x = torch.randn((3, 4, 5, 6, 7))
	a = x[:sequence_length, batch_size*idx:batch_size*(idx+1)]
	b = x[:sequence_length, batch_size*idx:batch_size*(idx+1), :, :, :]

	# print(a, "\n\n")
	# print(b)

	print(torch.equal(a, b))


def examine_pkl():
	with open('data.pkl', 'rb') as file:
		loaded_data = pickle.load(file)
	
	print(loaded_data[0])


def change_data_in_pickle():
	with open('train_obs_10.pickle', 'rb') as file:
		loaded_data = pickle.load(file)
		new_loaded_data = np.array(loaded_data)
		new_loaded_data = torch.from_numpy(new_loaded_data)
		d_shape = new_loaded_data.shape
		new_loaded_data = new_loaded_data.reshape((d_shape[1], d_shape[0], -1))
		new_loaded_data = new_loaded_data[:, :, :, None, None]
		file.close()

	with open('torch_train_obs_10.pickle', 'wb') as file:

		pickle.dump(new_loaded_data, file)


def create_h5_files():
	with open('torch_train_obs_10.pickle', 'rb') as file:
		data_10 = pickle.load(file)

		hf = h5py.File('torch_train_obs_10.h5', 'w')

		g_train = hf.create_group('training')
		g_train.create_dataset('features', data=data_10[:, :5])
		g_train.create_dataset('groups', data=data_10[:, :5])

		g_test = hf.create_group('validation')
		g_test.create_dataset('features', data=data_10[:, 5:])
		g_test.create_dataset('groups', data=data_10[:, 5:])

		hf.close()



def check_h5_file():
	hf = h5py.File("torch_train_obs_10.h5", 'r')

	print(hf["training"]["features"])





if __name__ == "__main__":

	# test_kl_loss_mvn()
	# print_h5_file()

	# slice_test()
	
	# examine_pkl()

	# change_data_in_pickle()

	# check_h5_file()

	# create_h5_files()
	
	
	pass
	
