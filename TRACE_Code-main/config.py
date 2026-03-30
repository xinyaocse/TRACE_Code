# RRDB
nf = 3
gc = 32


clamp = 2.0
channels_in = 3
ham = 1
lr = 1e-4
lr2 = 1e-3
lr_min = 1e-5
epochs = 201
weight_decay = 1e-5
init_scale = 0.01

# Train:

lamda_guide = 1
lamda_low_frequency = 1
lamda_fuse = 3

betas = (0.5, 0.999)
weight_step = 10
gamma = 0.9
# Inverse adversarial loss
beta = 0.01

pretrain = True

