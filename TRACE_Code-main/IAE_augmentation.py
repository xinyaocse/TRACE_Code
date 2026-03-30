import os
import glob

from torch.autograd import Variable
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torchvision
import warnings
from torchvision.utils import save_image
from args import get_args_parser
from util.utils import *
from model.model import *

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#####################
# Model initialize: #
#####################
args = get_args_parser()
tf = transforms.Compose([transforms.Resize((224, 224)),
                         transforms.ToTensor()]) 
checkpoint = torch.load("./checkpoints/{}/{}/{}_trained_checkpoint.pth".format(args.dataset, args.mode, args.model))
state_dict = checkpoint['model_state_dict']
model = torchvision.models.resnet50()
model.fc = nn.Linear(2048, 10)
model.load_state_dict(state_dict)
model.eval().to(device)
model_feature = torch.nn.Sequential(*list(model.children())[:-2])
model_feature.eval().to(device)

data_dir = "./{}/train_data/target_image".format(args.dataset)


file_paths = sorted(glob.glob(os.path.join(data_dir, '*')))

# select ten target image
paths = file_paths[:10]

train_data = torchvision.datasets.ImageFolder('./{}/train_data'.format(args.dataset), transform=tf)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False,
                                           pin_memory=True, drop_last=True)

# such an query image
cover = load_image(''./{}/train_data/query_image'.format(args.dataset)'').to(device)
cover_feature = model_feature(cover)
scores = []

# adversarial radius
xi = 8 / 255.0
# moving to the feature centeriod
if __name__ == '__main__':
    for i in range(10):
        # original target image
        target_image = load_image(paths[i]).to(device)
        # target feature
        target_feature = model_feature(target_image).to(device)
        perturbation = (torch.rand(1, 3, 224, 224).to(device) - 0.5) * 2 * xi
        perturbation = Variable(perturbation, requires_grad=True)
        optim = torch.optim.Adam([perturbation], lr=0.01)
        best_num = 0.0
        for i_epoch in range(100):
            scores = []
            for j, data in enumerate(train_loader):
                #
                image, label = data[0].to(device), data[1].to(device)
                power_feature = model_feature(target_image + perturbation).to(device)
                feature2 = model_feature(image).to(device)
                score = guide_loss(power_feature, feature2).to(device)
		coverscore = guide_loss(cover_feature, feature2).to(device)
                scores.append([label, score.item()])
		coverscores.append([label, coverscore.item()])
            scores.sort(key=lambda x: x[1])
            coverscores.sort(key=lambda x: x[1])
            power_vector = torch.zeros(1, args.k).to(device)
            target_vector = torch.ones(1, args.k).to(device)
            cover_vector = torch.ones(1, args.k).to(device)
            # retrieval list of top-10
            for index in range(args.k):
                labels, _ = scores[index]
                if labels == 1:
                    power_vector[0, index] = 1
                    cover_vector[0, index] = 1
            center_fea_loss = guide_loss(target_feature, power_feature).to(device)
            query_list_loss = NDCG(power_vector.tolist(), cover_vector.tolist(), args.k)
            target_list_loss = NDCG(power_vector.tolist(), target_vector.tolist(), args.k)
            # far away from query sample
            cover_fea_loss = guide_loss(power_feature, cover_feature).to(device)
            total_loss = center_list_loss + center_fea_loss - cover_fea_loss
            optim.zero_grad()
            perturbation.data = torch.clamp(perturbation.data, -xi, xi)
            total_loss.backward(retain_graph=True)
            optim.step()

            if power_vector.sum() > best_num:
                # save the best result
                best_num = power_vector.sum()
                best_result = target_image + perturbation.data
                save_image(best_result, args.IAE_path + str(i) + ".png")
            if best_num == 0 and i_epoch == 100:
                # iteration end
                inverse = target_image + perturbation.data
                save_image(inverse, args.IAE_path + str(i) + ".png")
            print(
                'Epoch [{}/{}], total_loss: {:.4f},center_fea_loss:{:.4f},cover_fea_loss:{:.4f}'.format(
                i_epoch + 1, 50, total_loss.item(), center_fea_loss.item(), cover_fea_loss.item()))
