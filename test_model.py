import os
import argparse
from tqdm.auto import tqdm
import copy
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from utils import cal_metric, load_dataset, change_new_state_dict

from networks.classifier import LinearClassifier
from networks.transfer import TransferModel
from networks.resnet_big import SupConResNet, SupCEResNet
from networks.inception import SupIncepResnet


def parse_args():
    parser = argparse.ArgumentParser('argument for testing')
    parser.add_argument('--data_path', type=str, help='data path to train the target model')
    parser.add_argument('--car_model', type=str, )
    parser.add_argument('--pretrained_model', type=str, default='supcon')
    parser.add_argument('--pretrained_path', type=str, help='path which stores the pretrained model weights')
    parser.add_argument('--window_size', type=int, default=29)
    parser.add_argument('--strided', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--trial_id', type=int, default=1)
    parser.add_argument('--ckpt', type=int, help='id checkpoint for pretrained model')
    args = parser.parse_args()
    
    if args.strided == None:
        args.strided = args.window_size
    return args

def load_models_weights(args, model, verbose=False):
    if verbose:
        print('Loading: ', model.__class__.__name__)
    if model.__class__.__name__ == 'LinearClassifier':
        model_file = f'{args.pretrained_path}/ckpt_class_epoch_{args.ckpt}.pth'
    else:
        model_file = f'{args.pretrained_path}/ckpt_epoch_{args.ckpt}.pth'
    ckpt = torch.load(model_file)
    state_dict = change_new_state_dict(ckpt['model'])
    model.load_state_dict(state_dict=state_dict)
    return model

    
def load_model(args, verbose=False, is_cuda=True):
    if args.pretrained_model == 'resnet':
        model = SupCEResNet(num_classes=5)
        model = load_models_weights(args, model, verbose)
    elif args.pretrained_model == 'supcon':
        supcon_model = SupConResNet(name='resnet18')
        classifier = LinearClassifier(n_classes=5, feat_dim=128)
        supcon_model = load_models_weights(args, supcon_model, verbose)
        classifier = load_models_weights(args, classifier, verbose)
        model = TransferModel(supcon_model.encoder, classifier)
    elif args.pretrained_model == 'incep':
        model = SupIncepResnet(num_classes=5)
        model = load_models_weights(args, model, verbose)
        
    if is_cuda:
        model = model.cuda()
    return model
        
def inference(model, data_loader, verbose=False):
    total_pred = np.empty(shape=(0), dtype=int)
    total_label = np.empty(shape=(0), dtype=int)
    
    model.eval()
    if verbose:
        data_loader = tqdm(data_loader)
    with torch.no_grad():
        for samples, labels in data_loader:
            samples = samples.cuda(non_blocking=True)
            outputs = model(samples)
            _, pred = outputs.topk(1, 1, True, True)
            pred = pred.t().cpu().numpy().squeeze(0)
            total_pred = np.concatenate((total_pred, pred), axis=0)
            total_label = np.concatenate((total_label, labels), axis=0)
            
    return total_label, total_pred
    
def evaluate(model, data_loader, verbose=False):
    total_label, total_pred = inference(model, data_loader, verbose=verbose) 
    _, results = cal_metric(total_label, total_pred)
    if verbose:
        for key, values in results.items():
            print(list("{0:0.4f}".format(i) for i in values))
    return results
                  
def test(args, verbose=False, is_cuda=True):
    train_loader, val_loader = load_dataset(args, trial_id=args.trial_id)
    model = load_model(args, is_cuda=is_cuda, verbose=verbose)
    results = evaluate(model, val_loader, verbose=verbose)
    return results
                  
if __name__ == '__main__':
    args = parse_args()
    test(args, verbose=True, is_cuda=True)