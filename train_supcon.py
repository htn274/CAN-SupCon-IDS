import os
import argparse 
import math
import numpy as np
from tqdm import tqdm
from datetime import datetime

from dataset import CANDataset
from utils import get_prediction, cal_metric, print_results
from networks.simple_cnn import SupConCNN, LinearClassifier

from SupContrast.util import set_optimizer, save_model
from SupContrast.util import AverageMeter
from SupContrast.util import adjust_learning_rate, warmup_learning_rate, accuracy
from SupContrast.losses import SupConLoss

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.backends.cudnn as cudnn
#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')

from sklearn.metrics import f1_score

NUM_CLASSES = 5


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--data_dir', type=str, help='directory of data for training')
    parser.add_argument('--model', type=str, help='choosing models in [inception, cnn]')
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--window_size', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--gpu_device', type=int, default=0)
    
    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    
    # optimization supcon
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    # optimization
    parser.add_argument('--learning_rate_classifier', type=float, default=0.1,
                        help='learning rate classifier')
    parser.add_argument('--lr_decay_epochs_classifier', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate_classifier', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay_classifier', type=float, default=0,
                        help='weight decay_classifier')
    parser.add_argument('--momentum_classifier', type=float, default=0.9,
                        help='momentum_classifier')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine') 
    opt = parser.parse_args()
    
    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu_device)
        
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
            
            
    opt.model_path = './save/{}/models/'
    opt.tb_path = './save/{}/runs/'
    current_time = datetime.now().strftime("%D_%H%M%S").replace('/', '')
    opt.model_name = 'SupCon_{}_lr{}_bs{}_{}'.format(opt.model, opt.learning_rate, opt.batch_size, current_time)
    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        
    opt.tb_folder = opt.model_path.format(opt.model_name)    
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder, exist_ok=True)
        
    opt.save_folder = opt.tb_path.format(opt.model_name)    
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
        num_workers=8, pin_memory=True)
    
    return train_loader, val_loader

def set_model(opt):
    model = SupConCNN(feat_dim=128)
    criterion_model = SupConLoss(temperature=opt.temp)
    classifier = LinearClassifier(n_classes=NUM_CLASSES)
    criterion_classifier = torch.nn.CrossEntropyLoss()
    
    if torch.cuda.is_available():
        #if torch.cuda.device_count() > 1:
        #    # for using multiple gpus
        #    model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion_model = criterion_model.cuda()
        classifier = classifier.cuda()
        criterion_classifier = criterion_classifier.cuda()
        # Incerease runtime performance
        cudnn.benchmark = True
    return model, criterion_model, classifier, criterion_classifier

def train_model(train_loader, model, criterion, optimizer, epoch, opt, logger, step):
    model.train()
    
    losses = AverageMeter()
    
    for idx, (images, labels) in tqdm(enumerate(train_loader)):
        step += 1
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        
        features = model(images)
        features = features.unsqueeze(1)
        loss = criterion(features, labels)
        
        losses.update(loss.item(), bsz)
        
        if step % opt.print_freq == 0:
            logger.add_scalar('loss_supcon', losses.avg, step)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return step, losses.avg


def train_classifier(train_loader, model, classifier, criterion, optimizer, epoch, opt, step, logger):
    model.eval()
    classifier.train()
    
    losses = AverageMeter()
    accs = AverageMeter()
    
    for idx, (images, labels) in tqdm(enumerate(train_loader)):
        step += 1
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        
        with torch.no_grad():
            features = model.encoder(images)
            
        output = classifier(features.detach())
        loss = criterion(output, labels)
        losses.update(loss.item(), bsz)
        acc = accuracy(output, labels, topk=(1, ))
        accs.update(acc[0], bsz)
        
        if step % opt.print_freq == 0:
            logger.add_scalar('loss_ce/train', losses.avg, step)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return step, losses.avg, accs.avg
    
    
def get_predict(outputs):
    _, pred = outputs.topk(1, 1, True, True)
    pred = pred.t().cpu().numpy().squeeze(0)
    return pred 

def validate(val_loader, model, classifier, criterion, opt):
    model.eval()
    classifier.eval()
    
    losses = AverageMeter()
    total_pred = np.array([], dtype=int)
    total_label = np.array([], dtype=int)
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.cuda(non_blocking=True)
            outputs = classifier(model.encoder(images))
            bsz = labels.shape[0]
            loss = criterion(outputs, labels.cuda())
            losses.update(loss.item(), bsz)
            
            pred = get_predict(outputs)
            total_pred = np.concatenate((total_pred, pred), axis=0)
            total_label = np.concatenate((total_label, labels), axis=0)
            
            
    f1 = f1_score(total_pred, total_label, average='weighted')
    return losses.avg, f1

def set_optimizer(opt, model, class_str=''):
    dict_opt = vars(opt)
    optimizer = optim.SGD(model.parameters(),
                    lr=dict_opt['learning_rate'+class_str],
                    momentum=dict_opt['momentum'+class_str],
                    weight_decay=dict_opt['weight_decay'+class_str])
    return optimizer

def adjust_learning_rate(args, optimizer, epoch, class_str=''):
    dict_args = vars(args)
    lr = dict_args['learning_rate'+class_str]
    if args.cosine:
        eta_min = lr * (dict_args['lr_decay_rate'+class_str] ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(dict_args['lr_decay_epochs'+class_str]))
        if steps > 0:
            lr = lr * (dict_args['lr_decay_rate'+class_str] ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    opt = parse_option()
    
    train_loader, val_loader = set_loader(opt)
    model, criterion_model, classifier, criterion_classifier = set_model(opt)
   
    optimizer_model = set_optimizer(opt, model)
    optimizer_classifier = set_optimizer(opt, classifier, '_classifier')
    
    logger = SummaryWriter(log_dir=opt.tb_folder, flush_secs=2)
    train_classifier_freq = 2
    step = 0
    
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer_model, epoch)
        
        new_step, loss = train_model(train_loader, model, criterion_model, optimizer_model, epoch, opt, logger, step)
        print('Epoch: {}, SupCon Loss: {:.4f}'.format(epoch, loss))
        # Train and validate classifier 
        if epoch % train_classifier_freq == 0:
            adjust_learning_rate(opt, optimizer_classifier, epoch // train_classifier_freq, '_classifier')
            new_step, loss_ce, train_acc = train_classifier(train_loader, model, classifier, criterion_classifier, optimizer_classifier, epoch, opt, step, logger)
            print('Classifier: Loss: {:.4f}, Acc: {}'.format(loss_ce, train_acc))
            loss, val_f1 = validate(val_loader, model, classifier, criterion_classifier, opt)
            logger.add_scalar('loss_ce/val', loss, step)
            print('Validation: Loss: {:.4f}, F1: {:.4f}'.format(loss, val_f1))

            
        step = new_step
        if epoch % opt.save_freq == 0:
            ckpt = 'ckpt_epoch_{}.pth'.format(epoch)
            save_file = os.path.join(opt.save_folder, ckpt)
            save_model(model, optimizer_model, opt, epoch, save_file)
            
    save_file = os.path.join(opt.save_folder, 'last.pth')
    save_model(model, optimizer_model, opt, opt.epochs, save_file)
    save_file = os.path.join(opt.save_folder, 'last_classifier.pth')
    save_model(classifier, optimizer_classifier, opt, opt.epochs, save_file)
        

if __name__ == '__main__':
    main()