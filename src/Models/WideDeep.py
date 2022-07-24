import torch
import torch.nn as nn
import torch.nn.functional as F

class WideDeep(nn.Module):
	def __init__(self,wide_dim, deep_dim, h_dim=[], sequential_model=True,act_window_size=30,bio_window_size=30):
		super(WideDeep,self).__init__()
		self.wide_dim = wide_dim
		self.deep_dim = deep_dim
		self.wide = nn.Linear(wide_dim,1,bias=True)
		self.act_window_size = act_window_size
		self.bio_window_size = bio_window_size

		self.deep = nn.Sequential()
		h_dim = [deep_dim] + h_dim
		for i in range(len(h_dim)-1):
			self.deep.add_module("deep-linear%d"%(i),nn.Linear(h_dim[i],h_dim[i+1]))
			self.deep.add_module("deep-relu%d"%(i),nn.ReLU())
			self.deep.add_module("deep-dropout%d"%(i),nn.Dropout(0.1))
		self.deep.add_module("deep-output",nn.Linear(h_dim[-1],1))

	def forward(self, music, user, time, environment,bio,activity,m1,m2):
		batch_size = music.shape[0]
		w_input = torch.cat((music,user),dim=1).float()
		if self.wide_dim == self.deep_dim:
			d_input = torch.cat((music,user),dim=1).float()
		else:
			d_input = torch.cat((music,user,time,environment,activity.view(batch_size,-1),bio.view(batch_size,-1)),dim=1).float()
		w_pred = self.wide(w_input)
		d_pred = self.deep(d_input)
		prediction = torch.sigmoid(w_pred + d_pred)
		return prediction*2+1, m1, m2