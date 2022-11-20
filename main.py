import torch
from torch import nn as nn
import torch_geometric.data as pygdat

from graphdata import GraphData
from graphnetNoU import GraphWrap
import numpy as np
import random

from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator

def plot_loss(loss_list,num_batch,num_epoch,epoch_step):
    x = list(range(num_batch))
    fig, ax = plt.subplots(1,1)
    ax.set_xlabel("epoch",fontsize = 20)
    ax.set_ylabel("loss",fontsize = 20)
	
    batch_per_epoch = num_batch//num_epoch
    ticks = list(range(0,num_batch,batch_per_epoch*epoch_step))
    labels = list(range(0,num_epoch,epoch_step))

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.plot(x,loss_list)

    plt.show()
    filename = "loss.png"
    fig.savefig(filename, bbox_inches='tight')
    plt.close()

def setup_seed(seed):
    torch.manual_seed(seed)          
    torch.cuda.manual_seed(seed)     
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    random_seed = 1
    setup_seed(random_seed)
    num_epochs = 500
    learning_rate = 0.001
    epoch_step = 50
    
    dataset = GraphData(1,1,False)
    train_idx = [0]
    test_idx = [0]
    train_set = torch.utils.data.Subset(dataset,train_idx)
    test_set = torch.utils.data.Subset(dataset,test_idx)
    train_loader = pygdat.DataLoader(train_set,batch_size=1,shuffle=True,num_workers=1)
    test_loader = pygdat.DataLoader(test_set,batch_size=1,shuffle=False)

    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    criterion = nn.MSELoss()
    model = GraphWrap(device,criterion,learning_rate,False)
    train_loss_list,total_batch_num = model.train(num_epochs,train_loader)
    model.test(test_loader)

    plot_loss(train_loss_list,total_batch_num,num_epochs,epoch_step)
if __name__ == '__main__':
    main()
