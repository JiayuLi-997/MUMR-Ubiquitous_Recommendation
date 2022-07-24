import numpy as np
import pandas as pd
import os
import argparse
import json
import gc
import random
import sys
import logging

import torch
from torch.utils.data import Dataset, DataLoader

class MoodDataset(Dataset):
    def __init__(self, exp_ids, act_window_size=10,bio_window_size=10, data_pth="../Features_add4/DictData/",
                    env_list="weather,GPS,time",wrist_suf="fill0",env_suf="",rating_level=5,
                    mhide_users=""):
        # load the dataset
        self.user = json.load(open(os.path.join(data_pth,"user.json")))
        self.music = json.load(open(os.path.join(data_pth,"music_norm.json")))
        self.lifelog = json.load(open(os.path.join(data_pth,"env_%s.json"%(env_suf))))
        self.wrist = np.load(os.path.join(data_pth,"wrist_%s.npy"%(wrist_suf)))
        self.label = json.load(open(os.path.join(data_pth,"label.json")))
        self.inter = json.load(open(os.path.join(data_pth,"interaction.json")))
        self.exp_ids = exp_ids
        self.act_window_size = act_window_size
        self.bio_window_size = bio_window_size
        if len(env_list):
            self.env_list = env_list.split(",")
        else:
            self.env_list = []
        if len(mhide_users):
            self.mhide_users=[int(u) for u in mhide_users.split(",")]
        else:
            self.mhide_users = []
        self.rating_level = rating_level
        # transfer rating to desired rating level
        for rec in self.label:
            self.label[rec][0] = self.r2s(self.label[rec][0])

    def r2s(self, r):
        # scale the ratings
        if self.rating_level==5:
            return (r-1)/4
        if self.rating_level==2:
            return int(r>3)
        if r>3:
            return 3
        if r<3:
            return 1
        return 2
    
    def __len__(self):
        return len(self.exp_ids)
    
    def __getitem__(self, idx):
        exp_id = self.exp_ids[idx]
        user_id, music_id = self.inter[str(exp_id)]
        music = np.array(self.music[str(music_id)])
        user = np.array(self.user[str(user_id)])
        time = np.array(self.lifelog[str(exp_id)]["time"])
        environment=[]
        environment = np.array([item for k in self.env_list for item in self.lifelog[str(exp_id)][k] ])
        activity = self.wrist[exp_id-1,-self.act_window_size:,1:]
        bio = self.wrist[exp_id-1,-self.bio_window_size:,:1]
        labels = np.array(self.label[str(exp_id)])
        if user_id in self.mhide_users:
            return [music,user,time,environment,bio,activity,labels,0]
        else:
            return [music,user,time,environment,bio,activity,labels,1]

