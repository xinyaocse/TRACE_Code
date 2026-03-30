import argparse


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--model', help='target classifiers,Resnet50 or AlexNet or Densenet121',
                        default='vgg16')
    parser.add_argument('--IAE_eps', type=float, help='', default=8 / 255)
    parser.add_argument('--dataset', type=float, help='', default="MNIST")
    parser.add_argument('--mode', help='', default="multiple")
    parser.add_argument('--eps', type=float, help=' budget', default=8 / 255)
    parser.add_argument('--k', type=int, help=' top-k', default=10)
    parser.add_argument('--IAE_path', help='', default=r'./MNIST/IAE/')
    parser.add_argument('--outputpath', help='', default=r'./MNIST/ensemble/')
    parser.add_argument('--pre_model', help='Init INN_model', default=r'./pretrained/model_final.pt')
    args = parser.parse_args()
    return args
