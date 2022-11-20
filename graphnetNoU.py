import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, LayerNorm
from torch_scatter import scatter_mean, scatter_add
from torch import nn
from torch_geometric.nn import MetaLayer
from torch_geometric.nn import MessagePassing

class ModifiedMetaLayer(MetaLayer):
    def __init___(self,edge_model,node_model):
        super().__init__(edge_model=edge_model,node_model=node_model)
        
    def forward( self, x, edge_index, edge_attr, batch ):
        row, col = edge_index

        edge_attr = self.edge_model(x[row], x[col], edge_attr, batch[row])
        x = self.node_model(x, edge_index, edge_attr, batch)

        return x, edge_attr

def CreateMLP(
    in_size,
    out_size,
    n_hidden,
    hidden_size,
    activation=nn.LeakyReLU,
    activate_last=False,
    layer_norm=False,
):
    arch = []
    l_in = in_size
    for l_idx in range(n_hidden):
        arch.append(Lin(l_in, hidden_size))
        arch.append(activation())
        l_in = hidden_size

    arch.append(Lin(l_in, out_size))

    if activate_last:
        arch.append(activation())

        if layer_norm:
            arch.append(LayerNorm(out_size))

    return Seq(*arch)

class EdgeModel(torch.nn.Module):
    def __init__(self,in_size, out_size, n_hidden, hidden_size, activation, activate_last, layer_norm):
        super().__init__()
        self.edge_mlp = CreateMLP(in_size, out_size, n_hidden, hidden_size, activation, activate_last, layer_norm)

    def forward(self, src, dest, edge_attr, batch):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr], 1)
        out = self.edge_mlp(out)
        return out

class NodeModel(torch.nn.Module):
    def __init__(self,in_size, out_size, n_hidden, hidden_size, activation, activate_last, layer_norm):
        super().__init__()
        self.node_mlp = CreateMLP(in_size, out_size, n_hidden, hidden_size, activation, activate_last, layer_norm)

    def forward(self, x, edge_index, edge_attr, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # batch: [N] with max entry B - 1.

        # official
        # x_i = x_i + Aggr(x_j, e_ij) 
        # row, col = edge_index
        # out = torch.cat([x[row], edge_attr], dim=1)
        # out = self.node_mlp_1(out)
        # out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        # out = torch.cat([x, out, u[batch]], dim=1)
        # out = self.node_mlp_2(out)

        # x_i = x_i + Aggr(e_ij)
        row, col = edge_index
        out = scatter_mean(edge_attr, row, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        out = self.node_mlp(out)
        return out

class Gcn4spmv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index, edge_weight) :
        # edge_index = torch.stack([edge_index[1],edge_index[0]],dim=0)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        return out

    def message(self, x_j, edge_weight) :
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class GraphNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        e_in = 1
        n_in = 1

        e_out = 32
        n_out = 32

        n_hidden = 2
        hidden_size = 32

        edge1 = EdgeModel(e_in+2*n_in,e_out,n_hidden,hidden_size,nn.ReLU,False,False)
        node1 = NodeModel(n_in+e_out,n_out,n_hidden,hidden_size,nn.ReLU,False,False)
        self.meta1 = ModifiedMetaLayer(edge_model = edge1,node_model=node1)

        edge2 = EdgeModel(e_out+2*n_out,e_out,n_hidden,hidden_size,nn.ReLU,False,False)
        node2 = NodeModel(n_out+e_out,n_out,n_hidden,hidden_size,nn.ReLU,False,False)
        self.meta2 = ModifiedMetaLayer(edge_model = edge2,node_model=node2)
        
        edge3 = EdgeModel(e_out+2*n_out,e_out,n_hidden,hidden_size,nn.ReLU,False,False)
        node3 = NodeModel(n_out+e_out,1,n_hidden,hidden_size,nn.ReLU,False,False)
        self.meta3 = ModifiedMetaLayer(edge_model = edge3,node_model=node3)

        self.spmv = Gcn4spmv()

    def forward(self,graph, flag=True):
        x, edge_attr = self.meta1(graph.x, graph.edge_index, graph.edge_weight, graph.batch)

        x, edge_attr = self.meta2(x, graph.edge_index, edge_attr, graph.batch)

        x, edge_attr = self.meta3(x, graph.edge_index, edge_attr, graph.batch)

        if flag:
            x = self.spmv(x, graph.edge_index, graph.edge_weight)
        
        return x

class GraphWrap:
    def __init__(self, device, criterion, learning_rate, is_float=True):
        if is_float:
            self.model = GraphNet()
        else:
            self.model = GraphNet().double()

        self.model = self.model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.schedule = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100,gamma=0.2)

    def train(self, num_epochs,trainloader):
        print('begin to train')
        self.model.train()
        i = 0
        train_loss_list = []
        for epoch in range(num_epochs):
            ## training step
            for graphs in trainloader:
                graphs = graphs.to(self.device)

                # loss = b-Ax
                out = self.model(graphs)
                loss = self.criterion(out, graphs.y)

                # loss = x-true_x
                # out = self.model(graphs,False)
                # loss = self.criterion(out, graphs.sol)

                self.optimizer.zero_grad()
                loss.backward()

                ## update model params
                self.optimizer.step()

                train_running_loss = loss.item()
            
                # if epoch % 50 == 0: 
                print('Epoch: {:3} | Batch: {:3}| Loss: {:6.4f} '.format(epoch,i,train_running_loss))

                i = i + 1
                train_loss_list.append(train_running_loss)

            self.schedule.step()

        return  train_loss_list, i

    def test(self, testloader):
        print('begin to test')
        self.model.eval()
        for graphs in testloader:
            graphs = graphs.to(self.device)
            with torch.no_grad():
                outputs = self.model(graphs,False)
                loss = self.criterion(outputs, graphs.sol)
                print('MSELoss = {}'.format(loss.item()))
                
