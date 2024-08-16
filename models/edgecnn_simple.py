import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout
from torch_geometric.nn import global_max_pool, BatchNorm
from spatial_edgecnn.spatial_edgecnn import AutomatedGraphDynamicEdgeConv
from temporal_edgecnn.temporal_edgecnn import TemporalSelfAttentionDynamicEdgeConv, TemporalDynamicEdgeConv, \
    AutomatedGraphDynamicEdgeConv, GeneralizedTemporalSelfAttentionDynamicEdgeConv, \
    GeneralizedTemporalSelfAttentionDynamicEdgeConvWithoutMask

import torch.nn as nn
from torch.autograd import Variable
import numpy as np

# Define the Multi-Layer Perceptron function
def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])

# Simplified Spatial Transformer Network (STN)
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()

        # Define convolutional layers for feature extraction
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        # Define fully connected layers for transformation matrix prediction
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        # Define batch normalization layers to stabilize learning
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # Activation function
        self.relu = nn.ReLU()

        # Dimension for the transformation matrix
        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]

        # Apply convolutional layers with ReLU and batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Apply max pooling and flatten the output
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # Apply fully connected layers to predict transformation matrix
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Create identity matrix as a baseline for transformation
        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()

        # Add identity matrix to the predicted matrix and reshape
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class Net(nn.Module):
    def __init__(self, out_channels, graph_convolution_layers=2, T=1, k=4, spatio_temporal_factor=0.01, aggr='max'):
        super().__init__()

        # Optional: Use STNkd for spatial transformation
        self.stn = STNkd(k = 2)  # Adjust `k` based on your feature dimension

        self.graph_convolution_layers = graph_convolution_layers

        # Define graph convolutional layers
        # The GeneralizedTemporalSelfAttentionDynamicEdgeConv is a custom graph convolutional layer 
        # that incorporates both spatial and temporal attention mechanisms
        self.conv1 = GeneralizedTemporalSelfAttentionDynamicEdgeConv(
            # nn takes a 4 dimensional input and produces a 64-dimensional features through 2 hidden layers of size 64
            nn                      = MLP([2*2, 32, 32]),
            attention_in_features   = 32,
            head_num                = 4,
            k                       = k,
            spatio_temporal_factor  = spatio_temporal_factor,
            T                       = T
        )
        self.conv2 = GeneralizedTemporalSelfAttentionDynamicEdgeConv(
            nn                      = MLP([2 * 32, 64]),
            attention_in_features   = 64,
            head_num                = 4,
            k                       = k,
            spatio_temporal_factor  = spatio_temporal_factor,
            aggr                    = aggr,
            T                       = T
        )

        # ********************* ARGUEMENT VARIABLE DEFINITIONS *********************
        # nn        : This is an instance of MLP (Multi-Layer Perceptron) which defines the network used for the graph convolution. 
        #           It consists of several fully connected layers. The input and output dimensions of these layers are provided


        # define fully connected layers that are used after the graph convolution layers to combine features. 
        # The number of input features to this layer depends on the number of graph convolution layers used
        # the first number is the amount of dimensions in the output features from each graphical conv layer
        if graph_convolution_layers == 2:
            self.lin1 = MLP([64 + 32, 512])
        elif graph_convolution_layers == 1:
            self.lin1 = MLP([32, 512])

        # final MLP, which further processes the MLP from self.lin1
        # 1. reduces dimensions
        # 2. dropout layer with rate out of 1
        # 3. reduces dimensions further
        # 4. another dropout layer with rate out of 1
        # 5. final linear layer that outputs the number of classes
        self.mlp = nn.Sequential(
        MLP([512, 256]), nn.Dropout(0.5), MLP([256, 128]), nn.Dropout(0.5),
        nn.Linear(128, out_channels)
        )

        # MORE SIMPLE
        #self.mlp = nn.Sequential(
        #MLP([512, 256]), nn.Dropout(0.5), nn.Linear(256, out_channels)
        #)

    def forward(self, data):
        sequence_numbers, pos, batch = data.x[:, 0].float(), data.pos.float(), data.batch

        # If using STNkd, apply spatial transformation
        pos = pos.reshape(len(torch.unique(data.batch)), -1, 2).transpose(2, 1)
        trans = self.stn(pos)
        pos = pos.transpose(2, 1)
        pos = torch.bmm(pos, trans)
        pos = pos.reshape(-1, 2)

        # Apply graph convolutions
        if self.graph_convolution_layers == 2:
            x1 = self.conv1(pos, sequence_numbers, batch)
            x2 = self.conv2(x1, sequence_numbers, batch)
            out = self.lin1(torch.cat([x1, x2], dim=1))
        else:
            x1 = self.conv1(pos, sequence_numbers, batch)
            out = self.lin1(x1)
        
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)