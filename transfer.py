import os
import argparse
from tqdm.auto import tqdm
import copy

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
from utils import cal_metric

from dataset import CANDataset
from networks.simple_cnn import SupConCNN
from SupContrast.networks.resnet_big import SupConResNet

from SupContrast.util import AverageMeter
from SupContrast.util import accuracy

T_NUM_CLASSES=4

class LinearClassifier(nn.Module):
    def __init__(self, n_classes, feat_dim):
        super().__init__()
        self.n_classes = n_classes
        self.fc = nn.Linear(feat_dim, n_classes)
        
    def forward(self, x):
        output = self.fc(x)
        return output
    
class TransferModel(nn.Module):
    def __init__(self, feat_extractor, classifier):
        super().__init__()
        self.encoder = copy.deepcopy(feat_extractor)
        self.classifier = copy.deepcopy(classifier)
        
    def forward(self, x):
        output = self.encoder(x)
        output = self.classifier(output)
        return output
    
def parse_args():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--pretrained_path', type=str)
    parser.add_argument('--car_model', type=str)
    parser.add_argument('--tf_algo', type=str) #'transfer', 'tune', 'transfer_tune'
    
    parser.add_argument('--window_size', type=int)
    parser.add_argument('--strided', type=int, default=None)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int)
    args = parser.parse_args()
    
    if args.strided == None:
        args.strided = args.window_size
    return args
    
def load_dataset(args, trial_id=1):
    data_dir = f'TFrecord_{args.car_model}_w{args.window_size}_s{args.strided}'
    data_dir = os.path.join(args.data_path, data_dir, str(trial_id))
    
    train_dataset = CANDataset(data_dir, window_size = args.window_size)
    val_dataset = CANDataset(data_dir, 
                            window_size = args.window_size,
                            is_train=False)

    train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, 
                shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=256, 
                num_workers=args.num_workers, pin_memory=True)
    
    return train_loader, val_loader

def change_new_state_dict_parallel(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    return new_state_dict

def load_source_model(args, ckpt_epoch, is_cuda=True):
    model_file = f'{args.pretrained_path}/ckpt_epoch_{ckpt_epoch}.pth'
    ckpt = torch.load(model_file)
    state_dict = change_new_state_dict_parallel(ckpt['model'])
    model = SupConResNet(name='resnet18')
    model.load_state_dict(state_dict=state_dict)
    if is_cuda:
        model = model.cuda()
    return model

def build_top_classifier(n_classes, feat_dim, lr=0.01, is_cuda=True):
    classifier = LinearClassifier(n_classes=n_classes, feat_dim=feat_dim)
    criterion = torch.nn.CrossEntropyLoss()

    if is_cuda:
        criterion = criterion.cuda()
        classifier = classifier.cuda()
        
    optimizer = optim.SGD(classifier.parameters(), lr=lr, 
                          momentum=0.9, weight_decay=0)
    return classifier, criterion, optimizer

def train_classifier(train_loader, model, classifier, criterion, optimizer, epoch):
    model.eval()
    classifier.train()
    
    losses = AverageMeter()
    accs = AverageMeter()
    
    for indx, (inputs, labels) in tqdm(enumerate(train_loader)):
        inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        
        with torch.no_grad():
            feats = model.encoder(inputs)
        
        outputs = classifier(feats.detach())
        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels, topk=(1, ))
        
        losses.update(loss.item(), bsz)
        accs.update(acc[0].item(), bsz)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return losses.avg, accs.avg

def train_whole(train_loader, model, criterion, optimizer, epoch):
    model.train()
    
    losses = AverageMeter()
    accs = AverageMeter()
    
    for indx, (inputs, labels) in tqdm(enumerate(train_loader)):
        inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        
        # print(inputs.shape)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels, topk=(1, ))
        
        losses.update(loss.item(), bsz)
        accs.update(acc[0].item(), bsz)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return losses.avg, accs.avg

def print_results(results):
    for key, values in results.items():
        print(key, list("{0:0.4f}".format(i) for i in values))

def evaluate(model, data_loader):
    total_pred = np.empty(shape=(0), dtype=int)
    total_label = np.empty(shape=(0), dtype=int)

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images = images.cuda(non_blocking=True)
            outputs = model(images)
            _, pred = outputs.topk(1, 1, True, True)
            pred = pred.t().cpu().numpy().squeeze(0)
            
            total_pred = np.concatenate((total_pred, pred), axis=0)
            total_label = np.concatenate((total_label, labels), axis=0)
            
    cm, results = cal_metric(total_label, total_pred)
    print_results(results)
    return results

       
def build_fine_tuned_model(source_model, classifier, is_cuda=True, lr=0.0001):
    fine_tuned_model = TransferModel(source_model.encoder, classifier)
    if is_cuda:
        fine_tuned_model = fine_tuned_model.cuda()
    optimizer = optim.SGD(fine_tuned_model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=0)
    return fine_tuned_model, optimizer

def do_helper(args, trial_id):
    train_loader, val_loader = load_dataset(args, trial_id=trial_id)
    source_model = load_source_model(args, ckpt_epoch=200)
    classifier, criterion, optimizer = build_top_classifier(n_classes=T_NUM_CLASSES, 
                                                           feat_dim=512, lr=0.0005)
    transfer = 'transfer' in args.tf_algo
    finetune = 'tune' in args.tf_algo
    # Train the classifier with a fixed pretrained model first
    if transfer:
        print('Training classifier ============')
        classifier_n_epochs = 30
        for epoch in range(1, classifier_n_epochs + 1):
            loss, acc = train_classifier(train_loader, source_model, classifier, 
                              criterion, optimizer, epoch)
            print(f'Epoch {epoch}: loss={loss}, acc={acc}')
        
    fine_tuned_model, optimizer = build_fine_tuned_model(source_model, classifier,
                                                        is_cuda=True, lr=0.0001)
    if transfer:
        print('Transfer Results')
        print('Evaluating on train set:')
        evaluate(fine_tuned_model, train_loader)
        print('Evaluating on test set:')
        results = evaluate(fine_tuned_model, val_loader)
        print('========================')
    
    if finetune:
        ft_n_epochs = 20
        for epoch in range(1, ft_n_epochs + 1):
            loss, acc = train_whole(train_loader, fine_tuned_model, 
                                    criterion, optimizer, epoch)
            print(f'Epoch {epoch}: loss={loss}, acc={acc}')

        print('Fine-tuning Results')
        print('Evaluating on train set:')
        evaluate(fine_tuned_model, train_loader)
        print('Evaluating on test set:')
        results = evaluate(fine_tuned_model, val_loader)
        
    return results
        
def main():
    args = parse_args()
    total_results = {}
    max_trials = 5
    for trial_id in range(1, max_trials + 1):
        results = do_helper(args, trial_id)
        for k, v in results.items():
            total_results.setdefault(k, [])
            total_results[k].append(v)
            
    print('Final results')
    total_results = {k: np.stack(v, axis=0) for k, v in total_results.items()}
    for k, v in total_results.items():
        print(k, list("{0:0.4f}".format(i) for i in v.mean(axis=0)))
    
if __name__ == '__main__':
    main()