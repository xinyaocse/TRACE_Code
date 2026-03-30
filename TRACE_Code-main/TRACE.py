import glob
import os
import torchvision

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import modules.Unet_common as common
import warnings
import time
from torch.autograd import Variable
from torchvision.utils import save_image
from args import get_args_parser
from util.utils import *
from model.model import *
import clip

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#####################
# Model initialize: #
#####################
args = get_args_parser()
INN_net = Model().to(device)
init_model(INN_net)
INN_net = torch.nn.DataParallel(INN_net, device_ids=[0])
para = get_parameter_number(INN_net)
print(para)
params_trainable = (list(filter(lambda p: p.requires_grad, INN_net.parameters())))
optim1 = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim1, c.weight_step, gamma=c.gamma)
optim_init = optim1.state_dict()
dwt = common.DWT()
iwt = common.IWT()

tf = transforms.Compose([transforms.Resize((224, 224)),
                         transforms.ToTensor()])

checkpoint = torch.load("./checkpoints/{}/{}/{}_trained_checkpoint.pth".format(args.dataset, args.mode, args.model))
state_dict = checkpoint['state_dict']
# number of dataset class
if args.dataset == "MNIST" or args.dataset == "CIFAR10":
    bits = 10
if args.dataset == "Oxford5k":
    bits = 11
if args.dataset == "Paris6k":
    bits = 12
# victim model
if args.model == "vgg16":
    if args.mode == 'binary':
        model = torchvision.models.vgg16()
        model.classifier[6] = nn.Linear(4096, 2)
    if args.mode == 'multiple':
        model = torchvision.models.vgg16()
        model.classifier[6] = nn.Linear(4096, bits)
    dimension = 512
    model.load_state_dict(state_dict)
    model.eval().to(device)
    vgg16_model_feature = torch.nn.Sequential(*list(model.children())[0])
    vgg16_model_feature.eval().to(device)
    print(vgg16_model_feature)

elif args.model == "alexnet":
    if args.mode == 'binary':
        model = torchvision.models.alexnet()
        model.classifier[-1] = nn.Linear(4096, 2)
    if args.mode == 'multiple':
        model = torchvision.models.alexnet()
        model.classifier[-1] = nn.Linear(4096, bits)
    dimension = 256
    model.load_state_dict(state_dict)
    model.eval().to(device)
    alexnet_model_feature = torch.nn.Sequential(*list(model.children())[0])
    alexnet_model_feature.eval().to(device)
    print(alexnet_model_feature)

elif args.model == "resnet50":
    if args.mode == 'binary':
        model = torchvision.models.resnet50()
        model.fc = nn.Linear(2048, 2)
    if args.mode == 'multiple':
        model = torchvision.models.resnet50()
        model.fc = nn.Linear(2048, bits)
    dimension = 2048
    model.load_state_dict(state_dict)
    model.eval().to(device)
    resnet50_model_feature = torch.nn.Sequential(*list(model.children())[:-2])
    resnet50_model_feature.eval().to(device)
    print(resnet50_model_feature)

elif args.model == "densenet121":
    if args.mode == 'binary':
        model = torchvision.models.densenet121()
        model.classifier = nn.Linear(1024, 2)
    if args.mode == 'multiple':
        model = torchvision.models.densenet121()
        model.classifier = nn.Linear(1024, bits)
    dimension = 1024
    model.load_state_dict(state_dict)
    model.eval().to(device)
    densenet121_model_feature = torch.nn.Sequential(*list(model.children())[:-1])
    densenet121_model_feature.eval().to(device)
    print(densenet121_model_feature)

elif args.model == "ViT"
    if args.mode == 'binary':
        model = torchvision.models.vit_l_32()
        model.classifier = nn.Linear(512, 2)
    if args.mode == 'multiple':
        model = torchvision.models.vit_l_32()
        model.classifier = nn.Linear(512, bits)
    dimension = 512
    model.load_state_dict(state_dict)
    model.eval().to(device)
    ViT_model_feature = torch.nn.Sequential(*list(model.children())[:-1])
    ViT_model_feature.eval().to(device)
    print(ViT_model_feature)

elif args.model == "clip_i2t":
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    dimension = 512
    model.eval().to(device)
    class CLIPImageFeature(nn.Module):
        def __init__(self, clip_model):
            super().__init__()
            self.clip_model = clip_model
        def forward(self, x):
            feat = self.clip_model.encode_image(x)
            feat = feat.float()
            feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat
    model_feature = CLIPImageFeature(model).to(device)
    model_feature.eval()
    print(model_feature)

elif args.model == "clip_t2i":
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    dimension = 512
    model.eval().to(device)
    class CLIPTextFeature(nn.Module):
        def __init__(self, clip_model):
            super().__init__()
            self.clip_model = clip_model

        def forward(self, text_tokens):
            feat = self.clip_model.encode_text(text_tokens)
            feat = feat.float()
            feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat
    model_feature = CLIPTextFeature(model).to(device)
    model_feature.eval()
    print(model_feature)

def clip_tokenize_text(text):
    if isinstance(text, str):
        text = [text]
    return clip.tokenize(text).to(device)

def get_clip_similarity(image_feat, text_feat):
    return image_feat @ text_feat.t()

# target image path
data_dir = args.IAE_path

paths = file_paths[:random.randint(0,9)]
# gallery retrieval
train_data = torchvision.datasets.ImageFolder('./{}/retrieval_set'.format{args.dataset}, transform=tf)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False, pin_memory=True, drop_last=True)
number = target_image.class

if __name__ == '__main__':
    totalTime = time.time()
    X_target = load_image(paths[0])
    X_target = Variable(X_target, requires_grad=True)
    optim2 = torch.optim.Adam([X_target], lr=c.lr2)
    if c.pretrain:
        load(args.pre_model, INN_net)
        optim1.load_state_dict(optim_init)
    # load original image
    data = load_image(ori_paths)
    cover = data.to(device)  # channels = 3
    cover_dwt_1 = dwt(cover).to(device)  # channels = 12
    cover_dwt_low = cover_dwt_1.narrow(1, 0, c.channels_in).to(device)  # channels = 3
    save_image(cover, args.outputpath + '{}/cover.png'.format(args.model))
    # RIEM augmentation
    for i_epoch in range(c.epochs):
        #################
        #             train:         #
        #################
        CGT = X_target.to(device)
        save_image(CGT, args.outputpath + '{}/CGT.png'.format(args.model))
        CGT_dwt_1 = dwt(CGT).to(device)  # channels =12
        CGT_dwt_low_1 = CGT_dwt_1.narrow(1, 0, c.channels_in).to(device)  # channels = 3
        input_dwt_1 = torch.cat((cover_dwt_1, CGT_dwt_1), 1).to(device)  # channels = 12*2
        output_dwt_1 = INN_net(input_dwt_1).to(device)  # channels = 24
        output_steg_dwt_2 = output_dwt_1.narrow(1, 0, 4 * c.channels_in).to(device)  # channels = 12
        output_step_low_2 = output_steg_dwt_2.narrow(1, 0, c.channels_in).to(device)  # channels = 3
        output_steg_dwt_low_1 = output_steg_dwt_2.narrow(1, 0, c.channels_in).to(device)  # channels = 3
        output_r_dwt_1 = output_dwt_1.narrow(1, 4 * c.channels_in, 4 * c.channels_in).to(device)
        output_steg_1 = iwt(output_steg_dwt_2).to(device)  # channels = 3
        output_r = iwt(output_r_dwt_1).to(device)
        output_steg_1 = torch.clamp(output_steg_1, min=0, max=1).to(device)
        eta = torch.clamp(output_steg_1 - cover, min=-args.eps, max=args.eps)
        output_steg_1 = torch.clamp(cover + eta, min=0, max=1)
        #################
        #             loss:           #
        #################
        g_loss = guide_loss(output_steg_1.cuda(), cover.cuda()).to(device)
        l_loss = guide_loss(output_step_low_2.cuda(), cover_dwt_low.cuda()).to(device)
        # loss_fn = torch.nn.MSELoss(reduction='mean')
        total_features = None
        for j in range(len(paths)):
            text_tokens = clip_tokenize_text(target_texts[j])
            ori_feature = model_feature(text_tokens)
            if total_features is None:
                total_features = ori_feature
            else:
                total_features = total_features + ori_feature
        # calculate average feature
        if total_features is not None:
            average_feature = total_features / len(paths)
        # calculate adversarial feature
        adv_feature = model_feature(output_steg_1).to(device)
        resnet50_loss = guide_loss(average_feature, adv_feature).to(device)
        DenseNet_loss = guide_loss(average_feature, adv_feature).to(device)
        AlexNet_loss = guide_loss(average_feature, adv_feature).to(device)
        # rank list loss
        scores = []
        queryscores = []
        for j, data in enumerate(train_loader):
            image, label = data[0].to(device), data[1].to(device)
            list_feature = model_feature(image).to(device)
            query_feature = model_feature(cover).to(device)
            score = guide_loss(list_feature, adv_feature).to(device)
            queryscore = guide_loss(list_feature, query_feature).to(device)
            scores.append([label, score.item()])
            queryscores.append([label, queryscore.item()])
        scores.sort(key=lambda x: x[1])
        vector1 = torch.zeros(1, args.k).to(device)
        query_vector=torch.zeros(1, args.k).to(device)
        for index in range(args.k):
            labels, _ = scores[index]
            if labels == number:
                vector1[0, index] = 1
        for index in range(args.k):
            labels, _ = queryscores[index]
            if labels == number:
	            query_vector[0, index] = 1
        vector2 = torch.ones(1, args.k).to(device)
        listloss = NDCG(vector1.tolist(), vector2.tolist(),args.k) - NDCG(vector1.tolist(), query_vector.tolist(),args.k)

        total_loss = c.lamda_guide * g_loss + c.lamda_low_frequency * l_loss \
                     + c.lamda_per * (0.3*resnet50_loss + 0.3*Alexnet_loss + 0.4*DenseNet_loss) - 0.3*listloss  
        #################
        #            Exit:            #
        #################
        if i_epoch >= c.epochs - 1:
            save_image(output_steg_1, args.outputpath + '{}/result.png'.format(args.model))
            output_r = normal_r(output_r)
            save_image(output_r, args.outputpath + '{}/r.png'.format(args.model))
            break
        #################
        #        Backward:      #
        #################
        optim1.zero_grad()
        optim2.zero_grad()
        total_loss.backward(retain_graph=True)
        optim1.step()
        optim2.step()

        weight_scheduler.step()
        lr_min = c.lr_min
        lr_now = optim1.param_groups[0]['lr']
        if lr_now < lr_min:
            optim1.param_groups[0]['lr'] = lr_min

        print(
            'Epoch [{}/{}], total_loss: {:.4f}, Alexnet_loss: {:.4f},DenseNet_loss:{:.4f},resnet50_loss:{:.4f},'
            'g_loss:{:.4f}'.format(
                i_epoch + 1, c.epochs, total_loss.item(), Alexnet_loss.item(), DenseNet_loss.item(), resnet50_loss.item(),
                g_loss.item()))