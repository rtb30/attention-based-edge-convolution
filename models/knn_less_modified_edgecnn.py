import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout
from torch_geometric.nn import global_max_pool, BatchNorm
from custom_graph_convolution import CGCNConv
from temporal_edgecnn.temporal_edgecnn import TemporalAutomatedGraphDynamicEdgeConv
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)).to(x.device)) \
            .view(1, 9).repeat(
            batchsize, 1)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class Net(torch.nn.Module):
    def __init__(self, out_channels, batch_size, num_points, num_frames, device, k=8, aggr='max'):
        super().__init__()
        self.stn = STN3d()
        # self.fstn = STNkd(k=64)
        self.conv1 = TemporalAutomatedGraphDynamicEdgeConv(nn_before_graph_creation=MLP([3, 16]),
                                                           nn=MLP([2 * 16, 64, 64, 64]),
                                                           graph_creation_in_features=16,
                                                           in_features=64,
                                                           head_num=4,
                                                           k=k,
                                                           batch_size=batch_size,
                                                           num_points=num_points,
                                                           num_frames=num_frames,
                                                           device=device,
                                                           aggr=aggr)
        self.conv2 = TemporalAutomatedGraphDynamicEdgeConv(nn_before_graph_creation=None,
                                                           nn=MLP([2 * 64, 128]),
                                                           graph_creation_in_features=64,
                                                           in_features=128,
                                                           head_num=8,
                                                           k=k,
                                                           batch_size=batch_size,
                                                           num_points=num_points,
                                                           num_frames=num_frames,
                                                           device=device,
                                                           aggr=aggr)

        # self.conv1 = TemporalSelfAttentionDynamicEdgeConv(MLP([2 * 3, 64, 64, 64]),
        #                                                   64, 4, k, aggr)
        # self.conv2 = TemporalSelfAttentionDynamicEdgeConv(MLP([2 * 64, 128]),
        #                                                   128, 8, k, aggr)
        self.lin1 = MLP([128 + 64, 1024])

        self.mlp = Seq(
            MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, out_channels))
        self.num_frames = num_frames

    def forward(self, data):
        sequence_numbers, pos, batch = data.x[:, 0].float(), data.pos.float(), data.batch
        sequence_numbers -= torch.min(sequence_numbers)
        sequence_numbers /= torch.max(sequence_numbers)
        pos = pos.reshape(len(torch.unique(data.batch)), -1, 3).transpose(2, 1)
        trans = self.stn(pos)
        pos = pos.transpose(2, 1)
        pos = torch.bmm(pos, trans)
        pos = pos.reshape(-1, 3)
        x1, first_edge_index = self.conv1(pos, sequence_numbers, batch)
        x2, second_edge_index = self.conv2(x1, sequence_numbers, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        unscaled_sequence_numbers = (self.num_frames - 1) * ((
                                                                     sequence_numbers - torch.min(sequence_numbers)
                                                             ) / (
                                                                     torch.max(sequence_numbers) - torch.min(
                                                                 sequence_numbers)
                                                             )) + 1
        return F.log_softmax(out, dim=1), torch.stack((first_edge_index, second_edge_index),
                                                      0), unscaled_sequence_numbers
