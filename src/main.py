import numpy as np
import pandas as pd
import os
import argparse
import json
import gc
import random
import sys
import logging

from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from Datasets.MoodDataset import MoodDataset
from Models.WideDeep import WideDeep
from Models.MUMR import *

logger = logging.getLogger()
INF = 100000000

class MultiLoss(nn.Module):
	def __init__(self,alpha,beta,gamma,):
		super(MultiLoss,self).__init__()
		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma
	def mse_loss(self, y,pred_y):
		return torch.pow(y-pred_y,2).sum()

	def forward(self, m1, m2, r, pred_m1, pred_m2, pred_r):
		m1_loss = self.mse_loss(m1,pred_m1)
		m2_loss = self.mse_loss(m2,pred_m2)
		# print("m2:", m2)
		# print("pred_m2: ",pred_m2)
		# print("mse_loss: ",m2_loss)
		loss = m1_loss*self.alpha + m2_loss*self.beta

		# inputs = "loss"
		# while inputs != 'continue':
		# 	try:
		# 		print(eval(inputs))
		# 	except Exception as e:
		# 		print(e)
		# 	inputs = input()
		
		r_loss = self.mse_loss(r,pred_r)
		loss += r_loss*self.gamma
		return loss, r_loss,m1_loss,m2_loss

def l2_reg_loss(model):
	"""Returns the squared L2 norm of output layer of given model"""
	tol_loss = 0.0
	for name, param in model.named_parameters():
		if "w_embedding" in name or "t_embedding" in name or "e_embedding" in name:
			tol_loss += torch.sum(torch.square(param))

	return tol_loss


def str2bool(v):
	return v.lower() in ('yes', 'true', 't', '1', 'y')

def add_argument(parser):
	parser.register('type','bool',str2bool)
	
	# Data loading
	parser.add_argument("--data_pth",type=str,default="../Features_add4/DictData")
	parser.add_argument("--split_pth",type=str,default="../Features_add4/split/")
	parser.add_argument("--wrist_suf",type=str,default="",help="Wrist suffix for filename.")
	parser.add_argument("--lifelog_suf",type=str,default="",help="Lifelog suffix for filename.")
	parser.add_argument("--env_list",type=str,default="weather,GPS")
	
	# Goal Selection
	parser.add_argument("--model",type=str,default="MUMR",help="Model to train.")
	parser.add_argument("--train_type",type=str,default="LOO",help="Train type: LOO, LOSO, CV(for CV10), or CV5")
	parser.add_argument("--rating_level",type=int,default=3,help="Rating level: 2,3,5")
	parser.add_argument("--true_m1",type='bool',default='False')
	parser.add_argument("--true_m2",type='bool',default='False')
	parser.add_argument("--input_mood",type='bool',default='True')
	parser.add_argument("--input_mood1",type='bool',default='True')
	parser.add_argument("--um_only",type='bool',default='False',help="Whether to use user and music information only.")

	# Saving settings
	parser.add_argument("--save_pth",type=str,default="../predict_results_add4/")
	parser.add_argument("--log_file",type=str,default="./checkpoints/")
	parser.add_argument("--checkpoint",type='bool',default='True')
	parser.add_argument("--cp_pth",type=str,default="./checkpoints/")
	parser.add_argument("--save_annotation",type=str,default="user,music")
	parser.add_argument("--result_anno",type=str,default="user,music")

	# Data parameters
	parser.add_argument("--act_window_size",type=int,default=20)
	parser.add_argument("--bio_window_size",type=int,default=0)

	# Training settings
	parser.add_argument("--gpu",type=int,default=0,help="gru id")
	parser.add_argument("--batch_size",type=int,default=16)
	parser.add_argument("--lr",type=float,default=1e-3)
	parser.add_argument("--l2",type=float,default=1e-6)
	parser.add_argument("--lifelog_l2",type=float,default=0.2)
	parser.add_argument("--max_epoch",type=int,default=1000)
	parser.add_argument("--patience",type=int,default=30)
	parser.add_argument("--alpha",type=float,default=0.0,help="weight for loss mood1.")
	parser.add_argument("--beta",type=float,default=0.0,help="weight for loss mood2.")
	parser.add_argument("--gamma",type=float,default=1.0,help="weight for loss gamma.")

	# Training parameters
	parser.add_argument("--mhide_users",type=str,default="") # list of users to hide mood information 
	parser.add_argument("--embed_dim",type=int,default=16)
	parser.add_argument("--hidden_dim",type=int,nargs="+",default=[256])
	parser.add_argument("--a_hdim",type=int,default=8)
	parser.add_argument("--b_hdim",type=int,default=8)
	parser.add_argument("--t_hdim",type=int,default=2)
	parser.add_argument("--e_hdim",type=int,default=4)
	parser.add_argument("--kernel_size",type=int,default=5)
	parser.add_argument("--stride",type=int,default=3)

	return parser

def set_all_seeds(seed):
	random.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

def train(train_dataset,test_dataset,args,prefix="test"):
	# set device
	torch.backends.cudnn.deterministic = True
	device = torch.device("cuda:%d"%(args.gpu) if torch.cuda.is_available() else "cpu")
	logger.info("Training with device %s"%(str(device)))
	
	# define data & loss
	if not args.model in args.cp_pth:
		args.cp_pth = os.path.join(args.cp_pth,args.model)
	os.makedirs(args.cp_pth,exist_ok=True)
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)  
	criterion = MultiLoss(alpha=args.alpha,beta=args.beta,gamma=args.gamma)
	
	# set network
	input_dim = []
	example = train_dataset.__getitem__(0)
	for k in range(len(example)):
		example[k] = torch.Tensor(example[k]).unsqueeze(0)
		input_dim.append(example[k].shape)
		print(k,example[k].shape)

	# TODO
	if args.model == "WideDeep":
		wide_dim = input_dim[0][1]+input_dim[1][1]
		if args.um_only:
			deep_dim = wide_dim 
		else:
			deep_dim = wide_dim + input_dim[2][1] + input_dim[3][1] + input_dim[4][1]*input_dim[4][2] + input_dim[5][1]*input_dim[5][2]
		net = WideDeep(wide_dim,deep_dim,h_dim=args.hidden_dim,act_window_size=args.act_window_size,bio_window_size=args.bio_window_size)
	else:
		if args.um_only:
			args.a_hdim, args.b_hdim, args.t_hdim, args.e_hdim = 0,0,0,0
		net = MUMRNet(input_dim[0][1],input_dim[1][1],input_dim[2][1],input_dim[3][1],input_dim[4][1],input_dim[5][-1], embed_dim=args.embed_dim, 
		h_dim=args.hidden_dim, e_hdim=args.e_hdim, t_hdim=args.t_hdim, b_hdim=args.b_hdim, a_hdim = args.a_hdim, b_kernel = args.kernel_size, b_stride= args.stride, device=device,
		m1_input = args.true_m1, m2_input = args.true_m2,act_window_size=args.act_window_size,bio_window_size=args.bio_window_size,input_mood=args.input_mood,
		input_mood1 = args.input_mood1)
	
	# initialize network
	logger.info(str(net))
	net.to(device)
	optimizer = optim.Adam(net.parameters(), lr=args.lr,weight_decay=args.l2)

	# run! 
	min_test_loss = INF
	earlystop_iters = 0
	stop_epoch = 0
	tol = 1e-4
	for epoch in range(args.max_epoch):
		train_loss = 0.0
		train_loss_r = 0.0
		net.train()
		for i, data in enumerate(train_loader, 0):
			music, user, time, env, bio, act, labels = [x.to(device) for x in data[:-1]]
			# users with mood label
			valid_mood = np.where(data[-1]==1)[0]
			# forward
			pred_r, pred_m1, pred_m2 = net(music,user,time,env,bio,act,labels[:,1:3],labels[:,3:])
			# loss calculation
			loss,loss_r, loss_m1, loss_m2 = criterion(labels[valid_mood,1:3], labels[valid_mood,3:], 
					labels[:,0].view(-1,1), pred_m1[valid_mood,:], pred_m2[valid_mood,:], pred_r)
			loss += l2_reg_loss(net)*args.lifelog_l2 # partially add l2 regularization
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			train_loss += loss.item()
			train_loss_r += loss_r.item()
		train_loss /= train_dataset.__len__()
		train_loss_r /= train_dataset.__len__()
		
		# prediction
		test_loss, test_loss_r, test_loss_m1, test_loss_m2 = 0.0, 0.0, 0.0, 0.0
		net.eval()
		with torch.no_grad():
			for i,data in enumerate(test_loader):
				music, user, time, env, bio, act, labels = [x.to(device) for x in data[:-1]]
				valid_mood = np.where(data[-1]==1)[0]
				pred_r, pred_m1, pred_m2 = net(music,user,time,env,bio,act,labels[:,1:3],labels[:,3:])
				optimizer.zero_grad()
				loss,loss_r,loss_m1,loss_m2 = criterion(labels[valid_mood,1:3], labels[valid_mood,3:], 
					labels[:,0].view(-1,1), pred_m1[valid_mood,:], pred_m2[valid_mood,:], pred_r)
				test_loss += loss.item()
				test_loss_r += loss_r.item()
				test_loss_m1 += loss_m1.item()
				test_loss_m2 += loss_m2.item()
		test_loss /= test_dataset.__len__()
		test_loss_r /= test_dataset.__len__()
		test_loss_m1 /= test_dataset.__len__()
		test_loss_m2 /= test_dataset.__len__()

		if test_loss < min_test_loss - tol:
			logger.info("Min loss from %.3f to %.3f"%(min_test_loss,test_loss))
			min_test_loss = test_loss
			earlystop_iters = 0
			stop_epoch = epoch
			logger.info("Save network to %s"%(os.path.join(args.cp_pth,"%s_best.pkl"%(prefix))))
			torch.save(net, os.path.join(args.cp_pth, "%s_best.pkl"%(prefix)))
		else:
			earlystop_iters += 1
		if earlystop_iters >args.patience:
			break
		if epoch% 5 ==0:
			logger.info("[epoch %d] Train loss: %.3f, Test loss: %.3f (m1: %.3f, m2: %.3f), Test Rating MSE: %.3f [%.3f] (earlystop: %d)"%(epoch,train_loss,test_loss, test_loss_m1,test_loss_m2,
			test_loss_r,train_loss_r,earlystop_iters))

def evaluate(train_dataset, test_dataset, args,prefix="test"):
	# load model
	net = torch.load(os.path.join(args.cp_pth,"%s_best.pkl"%(prefix)))
	# define data & loss
	device = torch.device("cuda:%d"%(args.gpu) if torch.cuda.is_available() else "cpu")
	if not args.model in args.cp_pth:
		args.cp_pth = os.path.join(args.cp_pth,args.model)
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)  

	results = []
	net.eval()
	choice_type = ["Test","Train"]
	for l_choice,loader in enumerate([test_loader,train_loader]):
		label_all = []
		prediction_all = []
		with torch.no_grad():
			for i,data in enumerate(loader):
				music, user, time, env, bio, act, labels = [x.to(device) for x in data[:-1]]
				if len(label_all) == 0:
					label_all = labels.cpu().numpy()
				else:
					label_all = np.concatenate((label_all,labels.cpu()))
				pred_r, pred_m1, pred_m2 = net(music,user,time,env,bio,act,labels[:,1:3],labels[:,3:])
				pred = np.concatenate((pred_r.cpu(),pred_m1.cpu(),pred_m2.cpu()),axis=1)
				if len(prediction_all):
					prediction_all = np.concatenate((prediction_all,pred))
				else:
					prediction_all = pred
		r_dir = "%s/%s/"%(args.cp_pth, args.save_annotation)
		os.makedirs(r_dir,exist_ok=True)
		if l_choice == 0:
			file = "train_"+prefix
		else:
			file = "test_"+prefix
		np.save( os.path.join(r_dir,file+"_pred.npy"),prediction_all)
		np.save( os.path.join(r_dir,file+"_label.npy"), label_all)
		prediction_all = torch.Tensor(prediction_all)
		label_all = torch.Tensor(label_all)
		label_r = label_all[:,0]
		prediction_r = prediction_all[:,0]
		mse = []
		for k in range(1,5):
			prediction_m = prediction_all[:,k]
			label_m = label_all[:,k]
			mse.append(((prediction_m-label_m)**2).mean().item())
		mse.append(((prediction_r-label_r)**2).mean().item())
		results += mse
		logger.info("%s Metrics: m1_x %.3f, m1_y %.3f, m2_x %.3f, m2_y %.3f, rating %.3f"%(choice_type[l_choice],*mse))
		logger.info("%s set shape: %s"%(choice_type[l_choice],str(prediction_all.shape)))

	return results



def get_mainargs(args):
	arg_dict = vars(args)
	main_list = []
	for k in arg_dict:
		if k in ["gpu","data_pth","split_pth","rating_level","model","save_pth","log_file","cp_pth",]:
			continue
		if k not in ["wrist_suf","lifelog_suf","env_list","save_annotation","result_anno"]:
			continue
		v = arg_dict[k]
		if type(v) == float:
			v = "%.6f"%(v)
		main_list.append(str(k)+"="+str(v))
	return ",".join(main_list)

def train_eval(args,model="MUMR"):
	all_seed = 0
	i = 0
	result_list = []
	while(os.path.exists("%s/%s/fold%d_train.npy"%(args.split_pth,args.train_type,i))):
		logger.info("Training with fold %d"%(i))

		# load data
		train_idx = np.load("%s/%s/fold%d_train.npy"%(args.split_pth,args.train_type,i))
		test_idx = np.load("%s/%s/fold%d_test.npy"%(args.split_pth,args.train_type,i))
		logger.info("Fold %d sample size -- train: %d, test: %d"%(i,train_idx.shape[0],test_idx.shape[0]))

		# control seeds
		if "LOO" in args.train_type:
			all_seed = i
		set_all_seeds(all_seed)
		i += 1
		# construct dataset
		train_dataset = MoodDataset(train_idx, act_window_size=args.act_window_size, bio_window_size=args.bio_window_size, data_pth=args.data_pth, env_list=args.env_list, wrist_suf=args.wrist_suf,lifelog_suf=args.lifelog_suf, rating_level=args.rating_level,
		mhide_users=args.mhide_users)
		test_dataset = MoodDataset(test_idx, act_window_size=args.act_window_size, bio_window_size=args.bio_window_size, data_pth=args.data_pth, env_list=args.env_list, wrist_suf=args.wrist_suf,lifelog_suf=args.lifelog_suf, rating_level=args.rating_level,
		mhide_users=args.mhide_users )

		# train and test
		train(train_dataset,test_dataset,args,prefix=get_mainargs(args)+"_fold%d"%(i))
		results = evaluate(train_dataset,test_dataset,args,prefix=get_mainargs(args)+"_fold%d"%(i))
		result_list.append(results)
		del train_dataset
		del test_dataset
		gc.collect()
 
	return result_list


def run(args):
	model = args.model
	result_list = train_eval(args,model=model)
	result_fold = pd.DataFrame(result_list,columns=["mse_m1x","mse_m1y","mse_m2x","mse_m2y","mse_rating",
							"mse_m1x_tr","mse_m1y_tr","mse_m2x_tr","mse_m2y_tr","mse_rating_tr",])
	result_fold["cmd"] = " ".join(sys.argv)
	columns = result_fold.columns
	agg_dict = {}
	for col in columns:
		if col == "cmd":
			continue
		if "tr" in col:
			agg_dict[col] = ["mean"]
		else:
			agg_dict[col] = ["mean",lambda x:list(x)]
	result_df = result_fold.groupby(["cmd"]).agg(agg_dict).reset_index()
	result_df.columns = [col[0] if "lambda" not in col[1] else col[0]+"_list" for col in result_df.columns]
	
	result_pth = os.path.join(args.save_pth,model)
	os.makedirs(result_pth,exist_ok=True)
	result_file = os.path.join(result_pth,"um_only%s_%s_%s%s.csv"%(str(args.um_only),args.result_anno,args.wrist_suf,args.lifelog_suf))
	logger.info("Save results to file: %s"%(result_file))
	result_exist = pd.DataFrame()
	if os.path.exists(result_file):
		result_exist = pd.read_csv(result_file)
	result_df = result_exist.append(result_df,ignore_index=True)
	result_df.to_csv(result_file,index=False)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser = add_argument(parser)
	args = parser.parse_args()
	if args.bio_window_size == 0:
		args.bio_window_size = args.act_window_size
		logger.info("Bio window size not defined, set to activity window size %d"%(args.act_window_size))

	logger.setLevel(logging.INFO)
	console = logging.StreamHandler()
	console.setLevel(logging.DEBUG)
	logger.addHandler(console)

	if args.log_file:
		os.makedirs(os.path.dirname(args.log_file),exist_ok=True)
	fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%Y/%m/%d %I:%M:%S %p')
	if args.checkpoint:
		logfile = logging.FileHandler(args.log_file, 'a')
	else:
		logfile = logging.FileHandler(args.log_file, 'w')
	logfile.setFormatter(fmt)
	logger.addHandler(logfile)
	logger.info('COMMAND: %s' % ' '.join(sys.argv))
	logger.info("Save log to file: %s"%(args.log_file))

	run(args)
