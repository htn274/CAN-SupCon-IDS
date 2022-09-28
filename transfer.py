import os
import argparse
from tqdm.auto import tqdm
import copy
import multiprocessing
from concurrent import futures

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
from utils import cal_metric

from dataset import CANDataset
from networks.inception import SupIncepResnet
from networks.simple_cnn import SupConCNN
from networks.classifier import LinearClassifier
from networks.transfer import TransferModel
from networks.resnet_big import SupConResNet, SupCEResNet

from supcon.util import AverageMeter, accuracy

#torch.cuda.set_device(1)

def parse_args():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--data_path', type=str, help='data path to train the target model')
    parser.add_argument('--pretrained_model', type=str, default='supcon') #resnet
    parser.add_argument('--pretrained_path', type=str, help='path which stores the pretrained model weights')
    parser.add_argument('--car_model', type=str, help='car model', default=None)
    parser.add_argument('--imprint', action='store_true')
    parser.add_argument('--tf_algo', type=str, help='[transfer, tune, transfer_tune, akc]', default='transfer') #'transfer', 'tune', 'transfer_tune'
    
    parser.add_argument('--num_classes', type=int, help='number of classes for target model')
    parser.add_argument('--window_size', type=int, default=29)
    parser.add_argument('--strided', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    
    parser.add_argument('--source_ckpt', type=int, help='id checkpoint for pretrained model')
    
    parser.add_argument('--lr_transfer', type=float, help='learning rate for transfer process', default=0.001)
    parser.add_argument('--lr_tune', type=float, help='learning rate for fine tuning process', default=0.00001)
    parser.add_argument('--transfer_epochs', type=int, default=30)
    parser.add_argument('--tune_epochs', type=int, default=10)
    parser.add_argument('--feat_dims', type=int, default=128)
    
    args = parser.parse_args()
    
    if args.strided == None:
        args.strided = args.window_size
    return args
    
def load_dataset(args, trial_id=1):
    if args.car_model is None:
        data_dir = f'TFrecord_w{args.window_size}_s{args.strided}'
    else:
        data_dir = f'TFrecord_{args.car_model}_w{args.window_size}_s{args.strided}'
    data_dir = os.path.join(args.data_path, data_dir, str(trial_id))
    
    train_dataset = CANDataset(data_dir,
                               window_size = args.window_size)
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
    model_dict = {
        'resnet': SupCEResNet(name='resnet18', num_classes=5),
        'supcon': SupConResNet(name='resnet18'),
        'incep': SupIncepResnet(num_classes=5)
    }
    model = model_dict[args.pretrained_model]
       
    if args.pretrained_path is not None:
        print("Loading pretrained model")
        model_file = f'{args.pretrained_path}/ckpt_epoch_{ckpt_epoch}.pth'
        ckpt = torch.load(model_file)
        state_dict = change_new_state_dict_parallel(ckpt['model'])
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
    
    for indx, (inputs, labels) in enumerate(train_loader):
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
    
    for indx, (inputs, labels) in enumerate(train_loader):
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

def evaluate(model, data_loader, is_print=False):
    total_pred = np.empty(shape=(0), dtype=int)
    total_label = np.empty(shape=(0), dtype=int)

    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.cuda(non_blocking=True)
            outputs = model(images)
            _, pred = outputs.topk(1, 1, True, True)
            pred = pred.t().cpu().numpy().squeeze(0)
            
            total_pred = np.concatenate((total_pred, pred), axis=0)
            total_label = np.concatenate((total_label, labels), axis=0)
            
    cm, results = cal_metric(total_label, total_pred)
    if is_print:
        print_results(results)
    return results

       
def build_fine_tuned_model(source_model, classifier, is_cuda=True, lr=0.0001):
    fine_tuned_model = TransferModel(source_model.encoder, classifier)
    if is_cuda:
        fine_tuned_model = fine_tuned_model.cuda()
    optimizer = optim.SGD(fine_tuned_model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=0)
    return fine_tuned_model, optimizer

def imprint(model, classifier, data_loader, num_class, device):
    print('Imprint the classifier of the model')
    model.eval()
    classifier.eval()
    feat_size = 128
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            # compute output
            output = model.encoder(inputs) 
            if batch_idx == 0:
                output_stack = output
                target_stack = targets
                feat_size=output.size(1)
            else:
                output_stack = torch.cat((output_stack, output), 0)
                target_stack = torch.cat((target_stack, targets), 0)
    
    new_weight = torch.zeros(num_class, feat_size).to(device)
    for i in range(num_class):
        tmp = output_stack[target_stack == i].mean(0)
        new_weight[i] = tmp / tmp.norm(p=2)
        
    classifier.fc.weight.data = new_weight
    return classifier

def do_helper(args, trial_id):
    train_loader, val_loader = load_dataset(args, trial_id=trial_id)
    source_model = load_source_model(args, ckpt_epoch=args.source_ckpt)
    classifier, criterion, optimizer =\
                                    build_top_classifier(n_classes=args.num_classes, 
                                                         feat_dim=args.feat_dims,
                                                         lr=args.lr_transfer)
    # print('Shape classifier: ', classifier.fc.weight.shape)
    if args.imprint:
        classifier = imprint(source_model, classifier, train_loader, num_class=4, device='cuda')
        
    #print('Shape classifier: ', classifier.fc.weight.shape)
    transfer = 'transfer' in args.tf_algo
    finetune = 'tune' in args.tf_algo
    # Train the classifier with a fixed pretrained model first
    if transfer:
        # print('Training classifier ============')
        transfer_epochs = args.transfer_epochs
        for epoch in range(1, transfer_epochs + 1):
            loss, acc = train_classifier(train_loader, source_model, classifier, 
                              criterion, optimizer, epoch)
            #print(f'Epoch {epoch}: loss={loss}, acc={acc}')
        
    fine_tuned_model, optimizer = build_fine_tuned_model(source_model, classifier,
                                                is_cuda=True, lr=args.lr_tune)
    if transfer:
        #print('Transfer Results')
        #print('Evaluating on train set:')
        evaluate(fine_tuned_model, train_loader)
        #print('Evaluating on test set:')
        results = evaluate(fine_tuned_model, val_loader)
        #print('========================')
    
    if finetune:
        tune_epochs = args.tune_epochs
        for epoch in range(1, tune_epochs + 1):
            loss, acc = train_whole(train_loader, fine_tuned_model, 
                                    criterion, optimizer, epoch)
            #print(f'Epoch {epoch}: loss={loss}, acc={acc}')
            if epoch % 5 == 0:
                results = evaluate(fine_tuned_model, train_loader, is_print=False)
                #print('Train f1: ', results['f1'].mean())
                results = evaluate(fine_tuned_model, val_loader, is_print=False)
                #print('Val f1: ', results['f1'].mean())
                
        #print('Fine-tuning Results')
        #print('Evaluating on train set:')
        train_res = evaluate(fine_tuned_model, train_loader)
        #print('Evaluating on test set:')
        val_res = evaluate(fine_tuned_model, val_loader)
        
    print('Finish : ', trial_id)
    return train_res, val_res
        
def update_results(results, total_results):
    for k, v in results.items():
        total_results.setdefault(k, [])
        total_results[k].append(v)
        
def print_total_results(results):
    results = {k: np.stack(v, axis=0) for k, v in results.items()}
    for k, v in results.items():
        print(k, list("{0:0.4f}".format(i) for i in v.mean(axis=0)))
        
def main_multi():
    num_cpu = multiprocessing.cpu_count()
    max_trials = 5
    workers = max(num_cpu, max_trials)
    
    args = parse_args()
    train_total_results = {}
    val_total_results = {}
    with futures.ProcessPoolExecutor(workers) as executor:
        to_do = []
        for trial_id in range(1, max_trials + 1):
            print('Submit : ', trial_id)
            future = executor.submit(do_helper, args, trial_id)
            to_do.append(future)

        total_results = {}
        for future in futures.as_completed(to_do):
            try:
                train_res, val_res = future.result()
                update_results(train_res, train_total_results)
                update_results(val_res, val_total_results)
            except Exception as error:
                print('An exception occurred: {}'.format(error))
                
    print('FINAL RESULTS')
    print('Train: ')
    print_total_results(train_total_results)
    print('Validation: ')
    print_total_results(val_total_results)

def main():
    args = parse_args()
    train_total_results = {}
    val_total_results = {}
    max_trials = 5
    for trial_id in range(1, max_trials + 1):
        train_res, val_res = do_helper(args, trial_id)
        update_results(train_res, train_total_results)
        update_results(val_res, val_total_results)
            
    print('FINAL RESULTS')
    print('Train: ')
    print_total_results(train_total_results)
    print('Validation: ')
    print_total_results(val_total_results)
    
if __name__ == '__main__':
    main_multi()