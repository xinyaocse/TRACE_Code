# args.py
# -*- coding: utf-8 -*-

import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(description="TRACE Attack Full Pipeline")

    parser.add_argument('--dataset', default='mnist', help='Dataset name')
    parser.add_argument('--model', default='alexnet', help='Model backbone')
    parser.add_argument('--mode', default='multiple')
    parser.add_argument('--workers', type=int, default=0)

    parser.add_argument('--IAE_eps', type=float, default=8 / 255.0)
    parser.add_argument('--IAE_path', default='./IAE_outputs')
    parser.add_argument('--target_imgs_dir', default='./target_images')
    parser.add_argument('--query_img', default='', help='path of query image')
    parser.add_argument('--m', type=int, default=50, help='Number of target samples')
    parser.add_argument('--lambda_j', type=float, default=0.3, help='Weight for H metric in IAE')

    parser.add_argument('--eps', type=float, default=8 / 255.0)
    parser.add_argument('--max_iter', type=int, default=200)
    parser.add_argument('--outputpath', default='./TRACE_outputs')
    parser.add_argument('--pre_model', default='./pretrained/model_final.pt')

    parser.add_argument('--substitute_dir', default='./checkpoints')

    parser.add_argument('--k', type=int, default=10, help='Top-k for retrieval')
    parser.add_argument('--eval_mode', default='', help='retrieval or classification')
    parser.add_argument('--visualize_tsne', action='store_true')
    parser.add_argument('--no_IAE', action='store_true', help='Disable IAE augmentation')
    parser.add_argument('--no_RIE', action='store_true', help='Disable RIE module')

    return parser
