import torch
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter
from torch.nn import BatchNorm1d as BN
from torch_cluster import knn


def make_proper_data(data, sequence_number, batch):
    source, source_batch, target, target_batch = data, batch, data.clone(), None
    index_mapper = torch.arange(0, len(data))
    batch_size = len(torch.unique(batch))
    frame_number = len(torch.unique(sequence_number))
    point_number = len(data) // (batch_size * frame_number)
    source_batch = (batch * frame_number + sequence_number - 1).long()
    target_batch = source_batch.clone()
    target = target.reshape(batch_size, frame_number, -1, data.shape[-1])
    index_mapper = index_mapper.reshape(batch_size, frame_number, -1, 1)
    target = target.repeat(frame_number, 1, 1, 1).reshape(batch_size, frame_number * frame_number, -1, data.shape[-1])
    index_mapper = index_mapper.repeat(frame_number, 1, 1, 1).reshape(batch_size, frame_number * frame_number, -1, 1)
    mask = torch.triu(torch.ones((frame_number, frame_number))).reshape(-1)
    target = target[:, mask == 1]
    index_mapper = index_mapper[:, mask == 1]
    target_batch = target_batch.reshape(-1, 1).repeat(1, frame_number).reshape(batch_size, -1, point_number)
    target_batch = target_batch[:, mask == 1]
    return source, source_batch, target.reshape(-1, data.shape[-1]), target_batch.reshape(-1), index_mapper.reshape(-1)


src = torch.arange(25, dtype=torch.float).reshape(1, 1, 5, 5).requires_grad_()  # 1 x 1 x 5 x 5 with 0 ... 25
indices = torch.tensor([[-1, -1], [.5, .5]], dtype=torch.float).reshape(1, 1, -1, 2)  # 1 x 1 x 2 x 2
output = F.grid_sample(src, indices)
print(src)
print(output)

a = torch.arange(30, dtype=torch.float64).reshape(10, 3)
indices = torch.arange(5, dtype=torch.long).reshape(-1, 1).repeat(1, 2).reshape(-1)
print(a)
print(indices)
print(scatter(a, indices, dim_size=5, dim=-2, reduce='max'))

A = torch.tensor([[0.2008, 0.0400, -0.0931, 1],
                  [0.2167, 0.0458, -0.1069, 2],
                  [0.1959, 0.0189, -0.0909, 30],
                  [-1.1217, -0.2696, 2.3543, 20],
                  [-0.0379, 0.0223, 0.1487, 14],
                  [-1.1447, -0.2898, 2.3234, 1]])

num_points = 10
num_frames = 10
batch_size = 10
x = torch.rand((batch_size * num_frames * num_points, 3))
sequence_number = torch.arange(1, num_frames + 1).reshape(1, -1, 1).repeat(batch_size, 1, num_points).reshape(-1)
batch = torch.arange(0, batch_size).reshape(-1, 1).repeat(1, num_points * num_frames).reshape(-1)
knn_input = torch.cat((x, sequence_number.reshape(-1, 1)), 1)
batch_norm = BN(4)
knn_input = batch_norm(knn_input)
knn_input[:, 3] *= 0.25
source_data, source_batch, target_data, target_batch, index_mapper = make_proper_data(knn_input, sequence_number, batch)

edge_index = knn(target_data, source_data, 2, target_batch, source_batch)
print(sequence_number[index_mapper[edge_index[1]]] - sequence_number[edge_index[0]])
print(edge_index[1] - edge_index[0])

