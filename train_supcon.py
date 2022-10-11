import os
import argparse 
import math
import numpy as np
from tqdm import tqdm
from datetime import datetime

from dataset import CANDataset
from utils import get_prediction, cal_metric, print_results
from networks.simple_cnn import SupConCNN
from networks.classifier import LinearClassifier

from networks.resnet_big import SupConResNet
from supcon.util import set_optimizer, save_model
from supcon.util import AverageMeter
from supcon.util import adjust_learning_rate, warmup_learning_rate, accuracy
from supcon.losses import SupConLoss

#from focalloss import FocalLoss

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.backends.cudnn as cudnn

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
    parser.add_argument('--rid', type=int, default=1)
    
    # temperature
    parser.add_argument('--temp', type=float, default=0.1,
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
    parser.add_argument('--epoch_start_classifier', type=int, default=50)
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
        
    opt.warm = False
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
    opt.model_name = f'SupCon_{opt.model}{opt.rid}_lr{opt.learning_rate}_{opt.learning_rate_classifier}_bs{opt.batch_size}_{opt.epochs}epoch_temp{opt.temp}_{current_time}'
    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        
    opt.tb_folder = opt.tb_path.format(opt.model_name)    
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder, exist_ok=True)
        
    opt.save_folder = opt.model_path.format(opt.model_name)    
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder, exist_ok=True)
        
        
    opt.log_file = f'./save/{opt.model_name}/log'
        
    return opt


def set_loader(opt):
    data_dir = f'{opt.data_dir}/{opt.rid}/'
    train_dataset = CANDataset(root_dir=data_dir, 
                               window_size=opt.window_size)
    val_dataset = CANDataset(root_dir=data_dir, 
                             window_size=opt.window_size,
                             is_train=False)
    #train_dataset.total_size = 100000
    #val_dataset.total_size = 10000
    #print('Train size: ', len(train_dataset))
    #print('Val size: ', len(val_dataset))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, 
        shuffle=True, num_workers=opt.num_workers,
        pin_memory=True, sampler=None)
    
    train_classifier_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, 
        shuffle=True, num_workers=opt.num_workers,
        pin_memory=True, sampler=None)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1024, shuffle=False,
        num_workers=2, pin_memory=True)
    
    return train_loader, train_classifier_loader, val_loader

def set_model(opt):
    #model = SupConCNN(feat_dim=128)
    model = SupConResNet('resnet18')
    criterion_model = SupConLoss(temperature=opt.temp, contrast_mode='one')
    classifier = LinearClassifier(n_classes=NUM_CLASSES, feat_dim=128)
    #class_weights = [0.25, 1.0, 1.0, 1.0, 1.0]
    #criterion_classifier = FocalLoss(gamma=0.0) #torch.nn.CrossEntropyLoss()
    criterion_classifier = torch.nn.CrossEntropyLoss()
    
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion_model = criterion_model.cuda()
        classifier = classifier.cuda()
        criterion_classifier = criterion_classifier.cuda()
        # Incerease runtime performance
        cudnn.benchmark = True
    print('Model device: ', next(model.parameters()).device)
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
        
        print('Data device: ', images.device)
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
        
        #warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        
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

optimize_dict = {
    'SGD' : optim.SGD,
    'RMSprop': optim.RMSprop,
    'Adam': optim.Adam
}

def set_optimizer(opt, model, class_str='', optim_choice='SGD'):
    dict_opt = vars(opt)
    optimizer = optimize_dict[optim_choice]
    if optim_choice == 'Adam':
        optimizer = optimizer(model.parameters(),
                    lr=dict_opt['learning_rate'+class_str],
                    weight_decay=dict_opt['weight_decay'+class_str])   
    else:
        optimizer = optimizer(model.parameters(),
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
        lr_decay_epochs_arr = list(map(int, dict_args['lr_decay_epochs'+class_str].split(',')))
        lr_decay_epochs_arr = np.asarray(lr_decay_epochs_arr)
        steps = np.sum(epoch > lr_decay_epochs_arr)
        if steps > 0:
            lr = lr * (dict_args['lr_decay_rate'+class_str] ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# python3 train_supcon.py --data_dir ../Data/TFRecord_w29_s15/1/ --batch_size 4096 --window_size 29 --cosine --print_freq 100 --mode cnn --gpu_device 0 --learning_rate 0.1 --learning_rate_classifier 0.5 --num_workers 4 --epochs 200 --save_freq 5

def main():
    opt = parse_option()
    
    train_loader, train_classifier_loader, val_loader = set_loader(opt)
    model, criterion_model, classifier, criterion_classifier = set_model(opt)
   
    optimizer_model = set_optimizer(opt, model, optim_choice='SGD')
    optimizer_classifier = set_optimizer(opt, classifier, class_str='_classifier', optim_choice='SGD')
    
    logger = SummaryWriter(log_dir=opt.tb_folder, flush_secs=2)
    #train_classifier_freq = 2
    step = 0
    
    log_writer = open(opt.log_file, 'w')
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer_model, epoch)
        
        new_step, loss = train_model(train_loader, model, criterion_model, optimizer_model, epoch, opt, logger, step)
        print('Epoch: {}, SupCon Loss: {:.4f}'.format(epoch, loss))
        log_writer.write('Epoch: {}, SupCon Loss: {:.4f}\n'.format(epoch, loss))
        # Train and validate classifier 
        class_epoch = epoch - opt.epoch_start_classifier + 1
        if class_epoch > 0:
            adjust_learning_rate(opt, optimizer_classifier, class_epoch, '_classifier')
            new_step, loss_ce, train_acc = train_classifier(train_classifier_loader, model, classifier, 
                                                            criterion_classifier, optimizer_classifier, epoch, opt, step, logger)
            print('Classifier: Loss: {:.4f}, Acc: {}'.format(loss_ce, train_acc))
            log_writer.write('Classifier: Loss: {:.4f}, Acc: {}\n'.format(loss_ce, train_acc))
            loss, val_f1 = validate(val_loader, model, classifier, criterion_classifier, opt)
            logger.add_scalar('loss_ce/val', loss, step)
            print('Validation: Loss: {:.6f}, F1: {:.8f}'.format(loss, val_f1))
            log_writer.write('Validation: Loss: {:.6f}, F1: {:.8f}\n'.format(loss, val_f1))

        step = new_step
        if epoch % opt.save_freq == 0:
            ckpt = 'ckpt_epoch_{}.pth'.format(epoch)
            save_file = os.path.join(opt.save_folder, ckpt)
            save_model(model, optimizer_model, opt, epoch, save_file)
            if class_epoch > 0:
                ckpt = 'ckpt_class_epoch_{}.pth'.format(epoch)
                save_file = os.path.join(opt.save_folder, ckpt)
                save_model(classifier, optimizer_classifier, opt, epoch, save_file)
            
    save_file = os.path.join(opt.save_folder, 'last.pth')
    save_model(model, optimizer_model, opt, opt.epochs, save_file)
    save_file = os.path.join(opt.save_folder, 'last_classifier.pth')
    save_model(classifier, optimizer_classifier, opt, opt.epochs, save_file)
        

# python3 train_supcon.py --data_dir ../Data/TFrecord_w29_s15/ --model resnet18 --save_freq 10 --window_size 29 --epochs 200 --num_workers 8 --temp 0.07 --learning_rate 0.1 --learning_rate_classifier 0.01 --cosine --epoch_start_classifier 170 --batch_size 1024 --rid 5 
if __name__ == '__main__':
    main()
