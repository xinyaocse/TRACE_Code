#config.py
# -*- coding: utf-8 -*-
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


lr = 2e-3 
iae_lr = 3e-3 

IAE_eps = 8.0/255.0 
eps = 8.0/255.0 

nf = 3
gc = 32
clamp = 2.0


lamda_per = 12.0             
lamda_low_frequency = 2.0    
lamda_j_default = 0.5        


init_scale = 0.01

m = 50  
num_targets_rie = 4  


default_z = 4  
default_k = 10  

use_binary_hash = True
