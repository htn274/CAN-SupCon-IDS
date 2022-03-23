import os
import argparse 
import math
import numpy as np
from tqdm import tqdm
from datetime import datetime

from dataset import CANDataset
from utils import get_prediction, cal_metric, print_results
from networks.inception import InceptionResnet
from networks.simple_cnn import BaselineCNNClassifier

from SupContrast.util import set_optimizer, save_model
from SupContrast.util import AverageMeter
from SupContrast.util import adjust_learning_rate, warmup_learning_rate, accuracy

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from sklearn.metrics import f1_score

NUM_CLASSES = 5
MODELS = {
   'inception': InceptionResnet,
    'cnn': BaselineCNNClassifier 
}

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--data_dir', type=str, help='directory of data for training')
    parser.add_argument('--model', type=str, help='choosing models in [inception, cnn]')
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--window_size', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--num_workers', type=int)
    
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.2,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    
    opt = parser.parse_args()
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate
            
            
    opt.model_path = './save/models/'
    opt.tb_path = './save/runs/'
    current_time = datetime.now().strftime("%D_%H%M%S")
    opt.model_name = '{}_lr{}_bs{}_{}'.format(opt.model, opt.learning_rate, opt.batch_size, current_time)
    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder, exist_ok=True)
        
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder, exist_ok=True)
        
    return opt


def set_loader(opt):
    train_dataset = CANDataset(root_dir=opt.data_dir, 
                               window_size=opt.window_size)
    val_dataset = CANDataset(root_dir=opt.data_dir, 
                             window_size=opt.window_size,
                             is_train=False)
    #train_dataset.total_size = 100000
    #val_dataset.total_size = 10000
    print('Train size: ', len(train_dataset))
    print('Val size: ', len(val_dataset))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, 
        shuffle=True, num_workers=opt.num_workers,
        pin_memory=True, sampler=None)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=0, pin_memory=True)
    
    return train_loader, val_loader

def set_model(opt):
    model = MODELS[opt.model]
    model = model(feat_dim=128, n_classes=NUM_CLASSES)
    # model = InceptionResnet(n_classes=NUM_CLASSES) 
    # model = BaselineCNN(n_classes=NUM_CLASSES)
    criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        #if torch.cuda.device_count() > 1:
        #    # for using multiple gpus
        #    model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        # Incerease runtime performance
        cudnn.benchmark = True
    return model, criterion

def train(train_loader, model, criterion, optimizer, epoch, opt, logger, step):
    model.train()
    
    losses = AverageMeter()
    accs = AverageMeter()
    
    for idx, (images, labels) in tqdm(enumerate(train_loader)):
        step += 1
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        
        output = model(images)
        loss = criterion(output, labels)
        losses.update(loss.item(), bsz)
        acc1 = accuracy(output, labels, topk=(1, ))
        accs.update(acc1[0], bsz)
        
        if step % opt.print_freq == 0:
            logger.add_scalar('loss/train', losses.avg, step)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return step, losses.avg, accs.avg
    
    
def get_predict(outputs):
    _, pred = outputs.topk(1, 1, True, True)
    pred = pred.t().cpu().numpy().squeeze(0)
    return pred 

def validate(val_loader, model, criterion, opt):
    model.eval()
    
    losses = AverageMeter()
    total_pred = np.array([], dtype=int)
    total_label = np.array([], dtype=int)
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.cuda(non_blocking=True)
            outputs = model(images)
            bsz = labels.shape[0]
            loss = criterion(outputs, labels.cuda())
            losses.update(loss.item(), bsz)
            
            pred = get_predict(outputs)
            total_pred = np.concatenate((total_pred, pred), axis=0)
            total_label = np.concatenate((total_label, labels), axis=0)
            
            
    f1 = f1_score(total_pred, total_label, average='weighted')
    return losses.avg, f1

def main():
    opt = parse_option()
    
    train_loader, val_loader = set_loader(opt)
    model, criterion = set_model(opt)
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    
    logger = SummaryWriter(log_dir=opt.tb_folder, flush_secs=2)
    step = 0
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)
        
        step, loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt, logger, step)
        print('Epoch: {}, Loss: {}, Acc: {}'.format(epoch, loss, train_acc))
        
        if epoch % 2 == 0:
            loss, val_f1 = validate(val_loader, model, criterion, opt)
            logger.add_scalar('loss/val', loss, step)
            print('Validation: Loss: {}, F1: {}'.format(loss, val_f1))

        if epoch % opt.save_freq == 0:
            ckpt = 'ckpt_epoch_{}.pth'.format(epoch)
            save_file = os.path.join(opt.save_folder, ckpt)
            save_model(model, optimizer, opt, epoch, save_file)
            
    save_file = os.path.join(opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)
        

if __name__ == '__main__':
    main()