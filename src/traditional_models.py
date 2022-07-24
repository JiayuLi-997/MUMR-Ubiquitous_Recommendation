import numpy as np
import pandas as pd
import os
import json
import gc
from datetime import datetime,timedelta
import random
import logging
import sys

from scipy.stats import f_oneway
from scipy import stats
from collections import Counter

from sklearn.ensemble import GradientBoostingRegressor as GBRT
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn import svm
from sklearn.ensemble import AdaBoostRegressor as AdaBoost
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import KFold
import argparse

logger = logging.getLogger()
def str2bool(v):
	return v.lower() in ('yes', 'true', 't', '1', 'y')

def add_argument(parser):
	parser.register('type','bool',str2bool)

	# Data loading
	parser.add_argument("--data_pth",type=str,default="../Features_add4/DictData")
	parser.add_argument("--split_pth",type=str,default="../Features_add4/split/")
	parser.add_argument("--use_wrist",type=int,default=1)
	parser.add_argument("--env_list",type=str,default="time,weather,GPS")
	parser.add_argument("--window",type=int,default=10)
	parser.add_argument("--wrist_suf",type=str,default="",help="Wrist suffix for filename.")
	parser.add_argument("--lifelog_suf",type=str,default="",help="Lifelog suffix for filename.")

	# Saving settings
	parser.add_argument("--save_pth",type=str,default="../predict_results_add4/")
	parser.add_argument("--log_file",type=str,default="./checkpoints/")
	parser.add_argument("--checkpoint",type='bool',default='True')
	parser.add_argument("--save_annotation",type=str,default="user,music")

	# Goal Selection
	parser.add_argument("--model",type=str,default="GBRT",help="Model to train.")
	parser.add_argument("--train_type",type=str,default="LOO",help="Train type: LOO, LOSO, CV(for CV10), or CV5")
	parser.add_argument("--rating_level",type=int,default=3,help="Rating level: 2,3,5")

	# Other params
	parser.add_argument("--model_params",type=str,default="{}",help="Model args as dict.")

	return parser

def r2s(r, rating_level):
	if rating_level==5:
		return (r-1)/4
	if rating_level==2:
		return int(r>3)
	if r>3:
		return 3
	if r<3:
		return 1
	return 2

def load_data(all_expids, window_size=10, data_pth="../Features_add4/DictData/", lifelog_suf="", wrist_suf="",use_wrist=0,env_list="time,GPS,weather",rating_level=3):
	label="rating"
	label_dict={"rating":0,"m1_x":1,"m1_y":2,"m2_x":3,"m2_y":4}

	# load saved data
	user = json.load(open(os.path.join(data_pth,"user.json")))
	music = json.load(open(os.path.join(data_pth,"music_ss.json")))
	lifelog = json.load(open(os.path.join(data_pth,"lifelog_ss%s.json"%(lifelog_suf))))
	wrist = np.load(os.path.join(data_pth,"wrist_ss%s.npy"%(wrist_suf)))
	labels = json.load(open(os.path.join(data_pth,"label.json")))
	inter = json.load(open(os.path.join(data_pth,"interaction.json")))
	for exp in labels:
		labels[exp][0] = r2s(labels[exp][0], rating_level)
	# construct samples
	X = []
	Y = []
	env_list = env_list.split(",")
	for expid in all_expids:
		r_x = []
		user_id, music_id = inter[str(expid)]
		r_x += music[str(music_id)] + user[str(user_id)]
		if use_wrist:
			# load wrist data
			r_x += wrist[int(expid)-1,-window_size:,:].reshape(-1).tolist()
		if env_list!=[""]:
			# load environment data
			r_x += [item for k in env_list for item in lifelog[str(expid)][k]]
		X.append(r_x)
		if label in label_dict:
			Y.append(labels[str(expid)][label_dict[label]]) 
		else:
			raise ValueError("Unknown label type: "+label) 
	return np.array(X),np.array(Y)


def train_eval(model_name, args):
	mse = [[],[]] # test mse list, train mse list
	i = 0
	all_seed = 0
	# load model parameters
	try:
		model_params = eval(args.model_params)
		if not type(model_params)==dict:
			print(a)
	except Exception as identifier:
		logger.warning("Load parameters error: ",identifier)
		logger.warning("Treat parameters as {}")
		model_params = {}
	logger.info("Model %s Parameters are %s"%(model_name,str(args)))

	while(os.path.exists("%s/%s/fold%d_train.npy"%(args.split_pth,args.train_type,i))):
		logger.info("Training with fold %d"%(i))

		# load data
		train_idx = np.load("%s/%s/fold%d_train.npy"%(args.split_pth,args.train_type,i))
		test_idx = np.load("%s/%s/fold%d_test.npy"%(args.split_pth,args.train_type,i))
		X_train, y_train = load_data(train_idx, window_size = args.window, data_pth=args.data_pth, lifelog_suf=args.lifelog_suf, wrist_suf = args.wrist_suf, use_wrist=args.use_wrist,env_list=args.env_list, rating_level=args.rating_level )
		X_test, y_test = load_data(test_idx, window_size = args.window, data_pth = args.data_pth, lifelog_suf=args.lifelog_suf, wrist_suf = args.wrist_suf, use_wrist=args.use_wrist,env_list=args.env_list, rating_level=args.rating_level )
		# set all seeds
		if args.train_type == "LOO":
			all_seed = i
		np.random.seed(all_seed)
		random.seed(all_seed)
		i += 1
		
		# Load and construct models
		if model_name=="LR":
			model = LogisticRegression(random_state=all_seed,max_iter=model_params.get("max_iter",1000)).fit(X_train, y_train)
		if model_name=="SVM":
			kernel = model_params.get("kernel","linear")
			C = model_params.get("C",1.0)
			if kernel=='linear':
				model = svm.SVR(C=C, kernel = kernel)
			else:
				gamma = model_params.get("gamma",'scale')
				if kernel in ["rbf","sigmoid"]:
					model = svm.SVR(C=C, kernel = kernel,gamma=gamma)
				if kernel =="poly":
					model = svm.SVR(C=C, kernel =kernel,gamma=gamma,degree=model_params.get("degree",3))
			model.fit(X_train, y_train)
		if model_name=="KNN":
			model = KNN(n_neighbors=model_params.get("neighbor",5), weights=model_params.get("weights","uniform"),
			leaf_size=model_params.get("leaf",30),p=model_params.get("p",2),).fit(X_train,y_train)
		if model_name=="RF":
			model = RF(random_state=all_seed,n_estimators=model_params.get("n_estimators",100),max_depth=model_params.get("depth",None),
						min_samples_split=model_params.get("split",2),min_samples_leaf=model_params.get("leaf",1),
						).fit(X_train,y_train)
		if model_name=="GBRT":
			if model_params.get("sample_weight",False):
				class_num = Counter(y_train)
				sample_weight = [class_num[y]/len(y_train) for y in y_train]
				model = GBRT(random_state=all_seed,learning_rate=model_params.get("lr",0.1),
						 n_estimators=model_params.get("n_estimators",100),subsample=model_params.get("subsample",1.0),
						 min_samples_split=model_params.get("split",2), min_samples_leaf=model_params.get("leaf",1),
						 max_depth=model_params.get("depth",3),max_features=model_params.get("feature",None),
						 n_iter_no_change=model_params.get("earlystop",None),
						).fit(X_train, y_train, sample_weight=sample_weight)
			else:
				model = GBRT(random_state=all_seed,learning_rate=model_params.get("lr",0.1),
						 n_estimators=model_params.get("n_estimators",100),subsample=model_params.get("subsample",1.0),
						 min_samples_split=model_params.get("split",2), min_samples_leaf=model_params.get("leaf",1),
						 max_depth=model_params.get("depth",3),max_features=model_params.get("feature",None),
						 n_iter_no_change=model_params.get("earlystop",None),
						).fit(X_train, y_train)
		if model_name == "AdaBoost":
			model = AdaBoost(n_estimators=model_params.get("n_estimators",50),learning_rate=model_params.get("lr",1.0),random_state=all_seed
						).fit(X_train,y_train)

		y_pred = model.predict(X_test)
		y_tr = model.predict(X_train)
		mse[0].append(((y_test-y_pred)**2).mean())
		mse[1].append(((y_train-y_tr)**2).mean())
		logging.info("Flod %d -- train MSE: %.3f, test MSE: %.3f, "%(i-1,mse[1][-1],mse[0][-1]))

	logging.info("train MSE: %.3f, test MSE: %.3f, "%(np.mean(mse[1]),np.mean(mse[0])))
	return mse


def run(args):
	# train and evaluate
	model = args.model
	mse_test, mse_train = train_eval(model,args)
	results = [[model, args.save_annotation, "rating", np.mean(mse_test), mse_test,np.mean(mse_train[1]),args]]

	# save results
	result_df = pd.DataFrame(results,columns=["model","feature","label","mse","mse_list",
							"mse_train","args"])
	result_pth = os.path.join(args.save_pth,model)
	os.makedirs(result_pth,exist_ok=True)
	result_file = os.path.join(result_pth,"%s_%s%s%s.csv"%(args.train_type,args.save_annotation,args.wrist_suf,args.lifelog_suf))
	logger.info("Save results to file: %s"%(result_file))
	result_exist = pd.DataFrame()
	if os.path.exists(result_file):
		result_exist = pd.read_csv(result_file)
	result_df = pd.DataFrame(results,columns=["model","feature","label","mse","mse_list",
							"auc_train","args"])
	result_df = result_exist.append(result_df,ignore_index=True)
	result_df.to_csv(result_file,index=False)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser = add_argument(parser)
	args = parser.parse_args()

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
	
