import torch
import torch.nn as nn
import torch.nn.functional as F

class MUMRNet(nn.Module):
	def __init__(self, m_dim, u_dim, t_dim, e_dim, b_dim, a_dim, 
				embed_dim=32, h_dim=[1024], e_hdim = 8, t_hdim=4,b_hdim=8, a_hdim=8,
				b_kernel = 5,b_stride=1,
				sequential_model=True,device="cpu",m1_input=False,m2_input=False,act_window_size=30,bio_window_size=30,input_mood=True,input_mood1=True):
		'''
		m1_input, m2_input: whether use the true mood as input
		h_dim: hiddend dims for predictors
		'''	
		super(MUMRNet, self).__init__()
		self.act_window_size = act_window_size
		self.bio_window_size = bio_window_size
		self.device = device
		self.m1_input=m1_input
		self.m2_input=m2_input
		self.e_hdim = e_hdim+t_hdim
		self.a_hdim = a_hdim
		self.b_hdim = b_hdim
		self.sequential_model = sequential_model
		self.input_mood = input_mood
		self.input_mood1 = input_mood1

		# set embeddings
		self.m_embedding = nn.Linear(m_dim,embed_dim)
		self.u_embedding = nn.Linear(u_dim,embed_dim)
		if self.e_hdim>0:
			self.e_embedding = nn.Linear(e_dim+t_dim,self.e_hdim)
		if self.a_hdim>0:
			self.a_embedding = nn.GRU(a_dim, a_hdim, num_layers=1,batch_first=True)
		if self.b_hdim>0:
			self.b_embedding = nn.Conv1d(in_channels=1, out_channels=1,kernel_size=b_kernel,stride=b_stride)
			self.b_hdim = int((b_dim -(b_kernel-1)-1)/b_stride+1)

		# set predictors
		l_hdim = self.e_hdim+self.a_hdim+self.b_hdim
		h_dim = [0] + h_dim
		# m1 predictor: user & context info as input
		self.m1_predictor = nn.Sequential()
		h_dim[0] = embed_dim+l_hdim
		for i in range(len(h_dim)-1):
			self.m1_predictor.add_module("m1_predictor-linear%d"%(i),nn.Linear(h_dim[i],h_dim[i+1]))
			self.m1_predictor.add_module("m1_predictor-relu%d"%(i),nn.ReLU())
		self.m1_predictor.add_module("m1_predictor-linear%d"%(len(h_dim)-1),nn.Linear(h_dim[-1],2))
		self.m1_predictor.add_module("m1_predictor-output",nn.Sigmoid())
		
		# m2 predictor: mood1 + user & context & music info as input
		self.m2_predictor = nn.Sequential()
		h_dim[0] = embed_dim*2+l_hdim+2
		for i in range(len(h_dim)-1):
			self.m2_predictor.add_module("m2_predictor-linear%d"%(i),nn.Linear(h_dim[i],h_dim[i+1]))
			self.m2_predictor.add_module("m2_predictor-relu%d"%(i),nn.ReLU())
		self.m2_predictor.add_module("m2_predictor-linear%d"%(len(h_dim)-1),nn.Linear(h_dim[-1],2))
		self.m2_predictor.add_module("m2_predictor-output",nn.Sigmoid())

		# rating predictor: mood2 + user & context & music info as inpuf
		self.r_predictor = nn.Sequential()
		h_dim[0] = embed_dim*2+l_hdim+1
		for i in range(len(h_dim)-1):
			self.r_predictor.add_module("r_predictor-linear%d"%(i),nn.Linear(h_dim[i],h_dim[i+1]))
			self.r_predictor.add_module("r_predictor-relu%d"%(i),nn.ReLU())
		self.r_predictor.add_module("r_predictor-linear%d"%(len(h_dim)-1),nn.Linear(h_dim[-1],1))
		self.r_predictor.add_module("r_predictor-output",nn.Sigmoid())
					   
	def forward(self, music, user, time, environment,bio,activity,m1,m2):
		batch_size = music.shape[0]
		m_emb = torch.relu(self.m_embedding(music.float()))
		u_emb = torch.relu(self.u_embedding(user.float()))
		if self.e_hdim>0:
			e_emb = torch.relu(self.e_embedding(torch.cat((environment,time),dim=1).float()))
			life_emb = e_emb
		if self.a_hdim>0:
			output, a_emb = self.a_embedding(activity.float(),torch.zeros((1,batch_size,int(self.a_hdim))).to(self.device))
			a_emb = a_emb.squeeze(0)
			if self.e_hdim>0:
				life_emb = torch.cat((life_emb,a_emb),dim=1)
			else:
				life_emb = a_emb
		if self.b_hdim>0:
			b_emb = self.b_embedding(bio.transpose(1,2).float())
			b_emb = b_emb.squeeze(1)
			if self.e_hdim+self.a_hdim>0:
				life_emb = torch.cat((life_emb,b_emb),dim=1)
			else:
				life_emb = b_emb
		
		if self.a_hdim+self.b_hdim+self.e_hdim>0:
			# with context
			u_c_emb = torch.cat((u_emb,life_emb),dim=1)
		else:
			# user & music only
			u_c_emb = u_emb
		
		mood1 = self.m1_predictor(u_c_emb)
		if self.m1_input:
			mood2 = self.m2_predictor(torch.cat((m_emb,u_c_emb,m1.float()),dim=1))
		elif self.input_mood1:
			mood2 = self.m2_predictor(torch.cat((m_emb,u_c_emb,mood1),dim=1))
		else:
			zero_mood = torch.zeros(batch_size,2).to(self.device)
			mood2 = self.m2_predictor(torch.cat((m_emb,u_c_emb,zero_mood),dim=1))
		if self.m2_input:
			rating = self.r_predictor(torch.cat((m_emb,u_c_emb,m2.float()[:,0].view(-1,1)),dim=1))
		elif self.input_mood:
			rating = self.r_predictor(torch.cat((m_emb,u_c_emb,mood2[:,0].view(-1,1)),dim=1))
		else:
			zero_mood = torch.zeros(batch_size).view(-1,1).to(self.device)
			rating = self.r_predictor(torch.cat((m_emb,u_c_emb,zero_mood),dim=1))
		
		return rating*2+1, mood1*2-1, mood2*2-1