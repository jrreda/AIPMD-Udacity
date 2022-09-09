# Train a new network on a data set with train.py
#
#     Basic usage: python train.py ./flowers 
#     Prints out training loss, validation loss, and validation accuracy as the network trains
#     Options:
#         Set directory to save checkpoints: python train.py ./flowers -s ../ImageClassifier/
#         Choose architecture: python train.py ./flowers -a vgg16 
#                              python train.py ./flowers -a vgg16 -s ../ImageClassifier/
#         Set hyperparameters: python train.py ./flowers -a vgg13 -lr 0.0001 -hu 2048 1024 -do 0.6 0.4 -e 2
#         Use GPU for training: python train.py ./flowers -a vgg13 -cuda True

import argparse
from torch import nn
from torch import optim
import os
import torch
import warnings
warnings.filterwarnings('ignore')
from model_functions import load_data, train_network, structure_network, save_checkpoint


if __name__ == '__main__':
    # Create the parser and add arguments
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument('data_dir', action="store", default="./flowers/", type=str)

    # optional arguments
    parser.add_argument('-s', '--save_dir', default="../ImageClassifier/", type=str)
    parser.add_argument('-a', '--arch',  help='pretrained model architecture', default="vgg13", type=str)
    parser.add_argument('-lr', '--lr', help='optimzer learning rate', default=0.001, type=float)
    parser.add_argument('-e', '--epochs', default=3, type=int)
    parser.add_argument('-hu', '--hidden_units', help="network hidden units", nargs='+', action='store', default=[4096, 1000], type=int)
    parser.add_argument('-do', '--dropouts', help="network dropout", nargs='+', action='store', default=[0.5, 0.3], type=float)
    parser.add_argument('-cuda', '--gpu', default="True", type=bool)

    # Parse and print the results
    args = parser.parse_args()

    # load and transform data
    data_transforms, image_datasets, dataloaders = load_data(args.data_dir)
    
    # define the model structure
    model = structure_network(args.arch, args.hidden_units, args.dropouts)
    
    # train the model
    optimizer = train_network(model, dataloaders, arch=args.arch, gpu=args.gpu, lr=args.lr, epochs=args.epochs)
    
    # save a checkpoint
    save_checkpoint(args.save_dir, model, optimizer, image_datasets, args.arch, args.hidden_units, args.dropouts)
