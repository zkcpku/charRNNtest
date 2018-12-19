
# 模型构建从这里开始
import torch as t
from torch.utils.data import DataLoader
from torch import nn
import visdom
import torchvision
import numpy as np
import torch.nn.functional as F
import os
import sys
import re
import time
import pickle
import random
# import time
from tensorboardX import SummaryWriter



# In[9]:


class Config(object):
	# 用来写训练参数
	use_gpu = True
	is_debug = False
	
	# 先是dataset中的参数
	data_path = 'data/charrnn_data.txt'
#     with open('tmp/charrnn_data.pickle','rb') as f:
#         data_list = pickle.load(f)
	with open('tmp/charrnn_dict.pickle','rb') as f:
		data_dict = pickle.load(f)
	char2id = data_dict[0]
	id2char = data_dict[1]
	n_step = 20
	
	# model中的参数
	vocab_size = len(char2id)
	embedding_dim = 512
	hidden_dim = 512
	num_layers = 2
	dropout = 0.5
	
	lr = 1e-3
	
	batch_size = 256 + 64
	
	epochs = 20
	
	model_save_path = 'tmp/CharRnn/'


# In[3]:


def f_id2char(list,opt):
	rst = [opt.id2char[int(e)] for e in list]
	return rst


# In[4]:


class CharRnnDataset(t.utils.data.Dataset):
	def __init__(self, opt):
		data_path = opt.data_path
		char2id = opt.char2id
		n_step = opt.n_step
		with open(data_path,'r',encoding = 'utf8') as f:
			data_list = f.readlines()
		if opt.is_debug:
			data_list = data_list[:500]
		data_list = [char2id[e] for line in data_list for e in line if e != '\n'] # 转为长列表，且汉字转数字
		num_seq = int(len(data_list) / n_step)# 限定每次喂入的序列长度（太长的话会梯度消失）
		# 存储数据矩阵的维度
		self.num_seq = num_seq
		self.n_step = n_step
		# 舍去最后几个不够一行的
		data_list = data_list[:num_seq * n_step]
		# 存成矩阵
		arr = np.array(data_list).reshape((num_seq,-1))
		self.arr = t.from_numpy(arr)
	def __getitem__(self, item):
		# 获取第item次的数据
		x = self.arr[item,:]
		# 将最后一个字符的输出 label 定为输入的第一个字符，也就是"床前明月光"的输出是"前明月光床"
		y = t.zeros(x.shape)
		y[:-1],y[-1] = x[1:],x[0]
		return x,y
	def __len__(self):
		return self.num_seq


# In[5]:




# 定义训练模型
class CharRnn(nn.Module):
	def __init__(self, opt):
		super(CharRnn, self).__init__()
		vocab_size = opt.vocab_size
		embedding_dim =  opt.embedding_dim
		hidden_dim = opt.hidden_dim
		num_layers = opt.num_layers
		dropout = opt.dropout
		
		self.use_gpu = opt.use_gpu
		self.hidden_dim = hidden_dim
		self.embedding_dim = opt.embedding_dim
		self.vocab_size = vocab_size
		self.num_layers = num_layers
		# word embedding层
		self.embeddings = nn.Embedding(vocab_size, embedding_dim)
		# lstm rnn层
		self.rnn = nn.LSTM(embedding_dim, self.hidden_dim, num_layers)
		# 线性层
		self.linear = nn.Linear(self.hidden_dim, vocab_size)
	
	def forward(self, input, hidden = None):
		num_seq, n_step = input.size()
		if hidden is None:
			h_0 = t.zeros((self.num_layers, n_step, self.hidden_dim)).float()
			c_0 = t.zeros((self.num_layers, n_step, self.hidden_dim)).float()
			# h_0 = input.data.new(self.num_layers, n_step, self.hidden_dim).fill_(0.)
			# c_0 = input.data.new(self.num_layers, n_step, self.hidden_dim).fill_(0.)
			if self.use_gpu:
				h_0 = h_0.cuda()
				c_0 = c_0.cuda()
		else:
			h_0,c_0 = hidden
		# 第一层:embedding
		embeddings = self.embeddings(input)# (num_seq, n_step, embeddings_dim)
		
		# 第二层:lstm
		output2, hidden = self.rnn(embeddings,(h_0,c_0))#(num_seq, n_step, hidden_dim)
		
		# 第三层:linear
		output3 = self.linear(output2.view(num_seq * n_step, -1))#(num_seq * n_step, vocab_size)
		return output3,hidden





def main():
	current_time = time.strftime("%Y-%m-%dT-H%HM%M", time.localtime())
	writer = SummaryWriter()

	USE_CUDA = t.cuda.is_available()
	device = t.device("cuda" if USE_CUDA else "cpu")
	print(device)

	opt = Config()

	# 试验一下Dataset
	train_set = CharRnnDataset(opt)



	train_data = DataLoader(train_set, opt.batch_size, True, num_workers=4)
	print(len(train_set))
	print(len(train_data))
	# print("help")
	# print(train_data)
	model = CharRnn(opt)
	if opt.use_gpu:
		model = model.cuda()

	criterion = nn.CrossEntropyLoss()

	optimizer = t.optim.Adam(model.parameters(), lr = opt.lr)


	# In[ ]:

	global_step = 1
	train_loss = 0
	for e in range(opt.epochs):
		try:
			train_loss = 0
			for data in train_data:
				x, y = data
				x = x.long()
				y = y.long()
				# print(x , y)
				if  opt.use_gpu:
					x = x.cuda()
					y = y.cuda()

				score, _ = model(x)
				
				loss = criterion(score,y.view(-1))
				
				optimizer.zero_grad()
				loss.backward()
				
				nn.utils.clip_grad_norm(model.parameters(), 5)
				optimizer.step()
				writer.add_scalar('perplexity' , np.exp(loss.item()), global_step) # 乘一个缩放，估计本轮困惑率
				writer.add_scalar('loss',loss.item(),global_step)
				global_step += 1
				train_loss += float(loss.item())

			print('epoch: {}, perplexity is: {:.3f}, lr:{:.1e}'.format(e+1, np.exp(train_loss / len(train_data)), opt.lr))

			if (e + 1) % 5 == 0:
				model_save_path = opt.model_save_path + current_time + 'with_epoch'+str(e+1) + '.model'
				print('\t',model_save_path)
				t.save(model.state_dict(),model_save_path)
		except Exception as ex:
			print("error with global_step",global_step)
			print(ex)
			model_save_path = opt.model_save_path + current_time + 'with_epoch'+str(e+1)+'with_step'+str(global_step) + '.model'
			print('\t',model_save_path)
			t.save(model.state_dict(),model_save_path)
			raise ex

	model_save_path = opt.model_save_path + current_time + '.model'
	print('\t',model_save_path)
	t.save(model.state_dict(),model_save_path)


if __name__ == '__main__':
	main()