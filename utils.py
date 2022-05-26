import os
import torch 
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from dataset import CANDataset

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
                val_dataset, batch_size=args.batch_size, 
                num_workers=args.num_workers, pin_memory=True)
    
    return train_loader, val_loader

def change_new_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    return new_state_dict

def plot_embeddings(embeddings, targets, xlim=None, ylim=None, save_dir=None):
    classes = ['Normal', 'DoS', 'Fuzzy', 'gear', 'RPM']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728','#9467bd',]
    n_classes = len(classes)
    plt.figure(figsize=(10,10))
    for i in range(n_classes):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(classes)
    
    if save_dir is not None:
        plt.savefig(save_dir, dpi=300)

def cal_metric(label, pred):
    cm = confusion_matrix(label, pred)
    recall = np.diag(cm) / np.sum(cm, axis = 1)
    precision = np.diag(cm) / np.sum(cm, axis = 0)
    f1 = 2*recall*precision / (recall + precision)
    
    total_actual = np.sum(cm, axis=1)
    true_predicted = np.diag(cm)
    fnr = (total_actual - true_predicted)*100/total_actual
                   
    return cm, {
    'fnr': np.array(fnr),
    'rec': recall,
    'pre': precision,
    'f1': f1
    }

def get_prediction(model, dataloader):
    with torch.no_grad():
        model.eval()
        prediction = np.zeros(len(dataloader.dataset))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            prediction[k:k+len(images)] = np.argmax(model(images).data.cpu().numpy(), axis=1)
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return prediction, labels

def draw_confusion_matrix(cm, classes, save_dir=None):
    cm_df = pd.DataFrame(cm,
                     index = classes, 
                     columns = classes)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm_df, annot=True, cmap='YlGnBu', cbar=False, linewidths=0.5)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    if save_dir is not None:
        plt.savefig(save_dir, dpi=300)
    plt.show()
    
def print_results(results):
    print('\t' + '\t'.join(map(str, results.keys())))
    for idx, c in enumerate(classes):
        res = [round(results[k][idx], 4) for k in results.keys()]
        output = [c] + res
        print('\t'.join(map(str, output)))
