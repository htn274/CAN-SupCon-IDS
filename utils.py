import torch 
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
cuda = torch.cuda.is_available()

classes = ['Normal', 'DoS', 'Fuzzy', 'gear', 'RPM']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728','#9467bd',]
n_classes = len(classes)

def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    for i in range(n_classes):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(classes)

def extract_embeddings(dataloader, model, emb_dims):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), emb_dims))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels

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

def cal_metric(label, pred):
    cm = confusion_matrix(label, pred)
    recall = np.diag(cm) / np.sum(cm, axis = 1)
    precision = np.diag(cm) / np.sum(cm, axis = 0)
    f1 = 2*recall*precision / (recall + precision)
    
    fnr = []
    total_samples = cm.sum(axis=1)
    for i in range(n_classes):
        fnr.append(100*(total_samples[i] - cm[i][i])/total_samples[i])
                   
    return cm, {
    'fnr': np.array(fnr),
    'rec': recall,
    'pre': precision,
    'f1': f1
    }

def draw_confusion_matrix(cm):
    cm_df = pd.DataFrame(cm,
                     index = classes, 
                     columns = classes)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    plt.show()
    
def print_results(results):
    print('\t' + '\t'.join(map(str, results.keys())))
    for idx, c in enumerate(classes):
        res = [round(results[k][idx], 4) for k in results.keys()]
        output = [c] + res
        print('\t'.join(map(str, output)))
