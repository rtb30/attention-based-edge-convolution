from typing import Callable, Union, Optional
from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor, Adj
import numpy as np
import torch
from torch import Tensor
from torch_multi_head_attention import MultiHeadAttention
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter
from torch_geometric.utils import softmax
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from automated_graph_creation.attention_for_graph_creation import SelfAttentionEdgeIndexCreatorLayer
from automated_graph_creation.temporal_attention_for_graph_creation import TemporalSelfAttentionEdgeIndexCreatorLayer
import math

try:
    from torch_cluster import knn
except ImportError:
    knn = None


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


class EdgeConv(MessagePassing):
    r"""The edge convolutional operator from the `"Dynamic Graph CNN for
    Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)}
        h_{\mathbf{\Theta}}(\mathbf{x}_i \, \Vert \,
        \mathbf{x}_j - \mathbf{x}_i),
    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* a MLP.
    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps pair-wise concatenated node features :obj:`x` of shape
            :obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, out_channels]`,
            *e.g.*, defined by :class:`torch.nn.Sequential`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"max"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, nn: Callable, aggr: str = 'max', **kwargs):
        super(EdgeConv, self).__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class TemporalDynamicEdgeConv(MessagePassing):
    def __init__(self, nn: Callable, k: int, aggr: str = 'max',
                 num_workers: int = 1, **kwargs):
        super(TemporalDynamicEdgeConv,
              self).__init__(aggr=aggr, flow='target_to_source', **kwargs)

        if knn is None:
            raise ImportError('`TemporalDynamicEdgeConv` requires `torch-cluster`.')

        self.nn = nn
        self.k = k
        self.num_workers = num_workers
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(
            self, x: Union[Tensor, PairTensor],
            sequence_number: Union[Tensor, PairTensor],
            batch: Union[OptTensor, Optional[PairTensor]] = None, ) -> Tensor:
        num_frames = len(np.unique(sequence_number.cpu().numpy()))
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        assert x[0].dim() == 2, \
            'Static graphs not supported in `TemporalDynamicEdgeConv`.'

        b: PairOptTensor = (None, None)
        if isinstance(batch, Tensor):
            # b = (batch, batch)
            b_list = [(batch * num_frames + sequence_number - 1).long(),
                      (batch * num_frames + sequence_number - 2).long()]
            b_list[1] = torch.where((sequence_number == 1) | (sequence_number == num_frames), b_list[0], b_list[1])
            b = (b_list[0], b_list[1])
        elif isinstance(batch, tuple):
            assert batch is not None
            b = (batch[0], batch[1])

        edge_index = knn(x[0], x[1], self.k, b[0], b[1],
                         num_workers=self.num_workers)

        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))

    def __repr__(self):
        return '{}(nn={}, k={})'.format(self.__class__.__name__, self.nn,
                                        self.k)


class TemporalAttentionDynamicEdgeConv(MessagePassing):
    def __init__(self, nn: Callable, gate_nn: Callable, k: int, aggr: str = 'max',
                 num_workers: int = 1, **kwargs):
        super(TemporalAttentionDynamicEdgeConv,
              self).__init__(aggr=aggr, flow='target_to_source', **kwargs)

        if knn is None:
            raise ImportError('`TemporalAttentionDynamicEdgeConv` requires `torch-cluster`.')

        self.nn = nn
        self.gate_nn = gate_nn
        self.k = k
        self.num_workers = num_workers
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)

    def forward(
            self, x: Union[Tensor, PairTensor],
            sequence_number: Union[Tensor, PairTensor],
            batch: Union[OptTensor, Optional[PairTensor]] = None, ) -> Tensor:
        num_frames = len(np.unique(sequence_number.cpu().numpy()))
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        assert x[0].dim() == 2, \
            'Static graphs not supported in `TemporalAttentionDynamicEdgeConv`.'

        b: PairOptTensor = (None, None)
        if isinstance(batch, Tensor):
            # b = (batch, batch)
            b_list = [(batch * num_frames + sequence_number - 1).long(),
                      (batch * num_frames + sequence_number - 2).long()]
            b_list[1] = torch.where(sequence_number == 1, b_list[0], b_list[1])
            b = (b_list[0], b_list[1])
        elif isinstance(batch, tuple):
            assert batch is not None
            b = (batch[0], batch[1])

        edge_index = knn(x[0], x[1], self.k, b[0], b[1],
                         num_workers=self.num_workers)

        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None, batch=batch)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))

    def aggregate(self, inputs: Tensor, index: Tensor,
                  batch: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        gate = self.gate_nn(inputs).view(-1, 1)
        gate = softmax(gate, index)
        # Apply attention mechanism
        return scatter(gate * inputs, index, dim=self.node_dim, dim_size=dim_size,
                       reduce=self.aggr)

    def __repr__(self):
        return '{}(nn={}, k={})'.format(self.__class__.__name__, self.nn,
                                        self.k)


class TemporalSelfAttentionDynamicEdgeConv(MessagePassing):
    def __init__(self, nn: Callable, in_features: int, head_num: int, k: int, aggr: str = 'max',
                 num_workers: int = 1, **kwargs):
        super(TemporalSelfAttentionDynamicEdgeConv,
              self).__init__(aggr=aggr, flow='target_to_source', **kwargs)

        if knn is None:
            raise ImportError('`TemporalSelfAttentionDynamicEdgeConv` requires `torch-cluster`.')

        self.nn = nn
        self.multihead_attn = MultiHeadAttention(in_features, head_num)
        self.k = k
        self.num_workers = num_workers
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.multihead_attn)
        reset(self.nn)

    def forward(
            self, x: Union[Tensor, PairTensor],
            sequence_number: Union[Tensor, PairTensor],
            batch: Union[OptTensor, Optional[PairTensor]] = None, ) -> Tensor:
        num_frames = len(np.unique(sequence_number.cpu().numpy()))
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        assert x[0].dim() == 2, \
            'Static graphs not supported in `TemporalSelfAttentionDynamicEdgeConv`.'

        b: PairOptTensor = (None, None)
        if isinstance(batch, Tensor):
            # b = (batch, batch)
            b_list = [(batch * num_frames + sequence_number - 1).long(),
                      (batch * num_frames + sequence_number - 2).long()]
            b_list[1] = torch.where(sequence_number == 1, b_list[0], b_list[1])
            b = (b_list[0], b_list[1])
        elif isinstance(batch, tuple):
            assert batch is not None
            b = (batch[0], batch[1])

        edge_index = knn(x[0], x[1], self.k, b[0], b[1],
                         num_workers=self.num_workers)

        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None, batch=batch)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))

    def aggregate(self, inputs: Tensor, index: Tensor,
                  batch: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        original_shape = inputs.shape
        # We assume K is fixed and the index tensor is sorted!
        attention_input_shape = list([int(original_shape[0] / self.k)]) + list(original_shape)
        attention_input_shape[1] = self.k
        self_attention_input = inputs.reshape(attention_input_shape)
        attn_output = self.multihead_attn(self_attention_input, self_attention_input, self_attention_input)
        attn_output = attn_output.reshape(original_shape)
        # Apply attention mechanism
        return scatter(attn_output, index, dim=self.node_dim, dim_size=dim_size,
                       reduce=self.aggr)

    def __repr__(self):
        return '{}(nn={}, k={})'.format(self.__class__.__name__, self.nn,
                                        self.k)
def count_GeneralizedTemporalSelfAttentionDynamicEdgeConv(m, x, y):
    x_input, sequence_number, batch = x
    frame_number = len(torch.unique(sequence_number))
    batch_size = len(torch.unique(batch))
    point_number = len(x_input) // (batch_size * frame_number)
    dim = x_input.shape[-1] + 1
    # KNN
    size_source = batch_size * point_number * (1 + ((frame_number * (frame_number-1))//2))
    size_target = batch_size * frame_number * point_number
    ops = (2*dim) * size_source * size_target
    m.total_ops += torch.DoubleTensor([ops])

# def count_DynamicEdgeConv(m, x, y):
#     x_input, batch = x
#     frame_number = len(torch.unique(sequence_number))
#     batch_size = len(torch.unique(batch))
#     point_number = len(x_input) // (batch_size * frame_number)
#     dim = x_input.shape[-1] + 1
#     # KNN
#     size_source = batch_size * point_number * (1 + ((frame_number * (frame_number-1))//2))
#     size_target = batch_size * frame_number * point_number
#     ops = (2*dim) * size_source * size_target
#     m.total_ops += torch.DoubleTensor([ops])

def count_Multi_head_self_attention(m, x, y):
    q, k, v = x
    q_dim = q.shape[-1] // m.head_num
    attention_ops = int((q_dim+1) * k.numel())  # + 1 for normalization
    query_ops = int(q.shape[1] * v.numel())

    # Softmax ops
    nfeatures = k.shape[1]
    batch_size = q.shape[0]* m.head_num * q.shape[1]
    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    softmax_ops = batch_size * (total_exp + total_add + total_div)
    m.total_ops += torch.DoubleTensor([attention_ops + softmax_ops + query_ops])

def make_proper_data(data, sequence_number, batch, self_loop=False, T=1):
    source, source_batch, target, target_batch = data, batch, data.clone(), None
    index_mapper = torch.arange(0, len(data), device=data.device)
    batch_size = len(torch.unique(batch))
    frame_number = len(torch.unique(sequence_number))
    point_number = len(data) // (batch_size * frame_number)
    source_batch = (batch * frame_number + sequence_number - 1).long()
    target_batch = source_batch.clone()
    target = target.reshape(batch_size, 1, frame_number, -1, data.shape[-1])
    index_mapper = index_mapper.reshape(batch_size, 1, frame_number, -1, 1)
    target = target.repeat(1, frame_number, 1, 1, 1).reshape(batch_size, frame_number * frame_number, -1, data.shape[-1])
    index_mapper = index_mapper.repeat(1, frame_number, 1, 1, 1).reshape(batch_size, frame_number * frame_number, -1, 1)
    if self_loop:
        mask = torch.tril(torch.ones((frame_number, frame_number), device=data.device))
    else:
        mask = torch.tril(torch.ones((frame_number, frame_number), device=data.device), diagonal=-1)
        mask[0][0] = 1
    tmp = torch.tril(torch.ones((frame_number, frame_number), device=data.device), diagonal=-T - 1)
    mask -= torch.tril(torch.ones((frame_number, frame_number), device=data.device), diagonal=-T - 1)
    mask = mask.reshape(-1)
    target = target[:, mask == 1]
    index_mapper = index_mapper[:, mask == 1]
    target_batch = target_batch.reshape(-1, 1).repeat(1, frame_number).reshape(batch_size, -1, point_number)
    target_batch = target_batch[:, mask == 1]
    return source, source_batch, target.reshape(-1, data.shape[-1]), target_batch.reshape(-1), index_mapper.reshape(-1)


class GeneralizedTemporalSelfAttentionDynamicEdgeConv(MessagePassing):
    def __init__(self, nn: Callable, T: int, attention_in_features: int, head_num: int, k: int,
                 aggr: str = 'max',
                 num_workers: int = 1, spatio_temporal_factor: float = 0, **kwargs):
        super(GeneralizedTemporalSelfAttentionDynamicEdgeConv,
              self).__init__(aggr=aggr, flow='target_to_source', **kwargs)

        if knn is None:
            raise ImportError('`GeneralizedTemporalSelfAttentionDynamicEdgeConv` requires `torch-cluster`.')

        self.nn = nn
        self.multihead_attn = MultiHeadAttention(attention_in_features, head_num)
        self.k = k
        self.num_workers = num_workers
        self.spatio_temporal_factor = spatio_temporal_factor
        self.T = T
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.multihead_attn)
        reset(self.nn)

    def forward(
            self, x: Union[Tensor, PairTensor],
            sequence_number: Union[Tensor, PairTensor],
            batch: Union[OptTensor, Optional[PairTensor]] = None, ) -> Tensor:
        knn_input = torch.cat((x, sequence_number.reshape(-1, 1)), 1)
        knn_input -= knn_input.min(0, keepdim=True)[0]
        knn_input /= knn_input.max(0, keepdim=True)[0]
        knn_input[:, -1] *= self.spatio_temporal_factor * math.sqrt(x.shape[-1])
        source_data, source_batch, target_data, target_batch, index_mapper = make_proper_data(data=knn_input,
                                                                                              sequence_number=sequence_number,
                                                                                              batch=batch,
                                                                                              self_loop=False,
                                                                                              T=self.T)
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        assert x[0].dim() == 2, \
            'Static graphs not supported in `GeneralizedTemporalSelfAttentionDynamicEdgeConv`.'
        edge_index = knn(target_data, source_data, self.k, target_batch, source_batch,
                         num_workers=self.num_workers)
        edge_index[1] = index_mapper[edge_index[1]]
        return self.propagate(edge_index, x=x, size=None, batch=batch)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))

    def aggregate(self, inputs: Tensor, index: Tensor,
                  batch: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        original_shape = inputs.shape
        # We assume K is fixed and the index tensor is sorted!
        attention_input_shape = list([int(original_shape[0] / self.k)]) + list(original_shape)
        attention_input_shape[1] = self.k
        self_attention_input = inputs.reshape(attention_input_shape)
        attn_output = self.multihead_attn(self_attention_input, self_attention_input, self_attention_input)
        attn_output = attn_output.reshape(original_shape)
        # Apply attention mechanism
        return scatter(attn_output, index, dim=self.node_dim, dim_size=dim_size,
                       reduce=self.aggr)

    def __repr__(self):
        return '{}(nn={}, k={})'.format(self.__class__.__name__, self.nn,
                                        self.k)


class GeneralizedTemporalSelfAttentionDynamicEdgeConvWithoutMask(MessagePassing):
    def __init__(self, nn: Callable, T: int, attention_in_features: int, head_num: int, k: int,
                 aggr: str = 'max',
                 num_workers: int = 1, spatio_temporal_factor: float = 0, **kwargs):
        super(GeneralizedTemporalSelfAttentionDynamicEdgeConvWithoutMask,
              self).__init__(aggr=aggr, flow='target_to_source', **kwargs)

        if knn is None:
            raise ImportError('`GeneralizedTemporalSelfAttentionDynamicEdgeConvWithoutMask` requires `torch-cluster`.')

        self.nn = nn
        self.multihead_attn = MultiHeadAttention(attention_in_features, head_num)
        self.k = k
        self.num_workers = num_workers
        self.spatio_temporal_factor = spatio_temporal_factor
        self.T = T
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.multihead_attn)
        reset(self.nn)

    def forward(
            self, x: Union[Tensor, PairTensor],
            sequence_number: Union[Tensor, PairTensor],
            batch: Union[OptTensor, Optional[PairTensor]] = None, ) -> Tensor:
        knn_input = torch.cat((x, sequence_number.reshape(-1, 1)), 1)
        knn_input -= knn_input.min(0, keepdim=True)[0]
        knn_input /= knn_input.max(0, keepdim=True)[0]
        knn_input[:, -1] *= self.spatio_temporal_factor * math.sqrt(x.shape[-1])
        # source_data, source_batch, target_data, target_batch, index_mapper = make_proper_data(knn_input,
        #                                                                                       sequence_number,
        #                                                                                       batch,
        #                                                                                       T=self.T)
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        assert x[0].dim() == 2, \
            'Static graphs not supported in `GeneralizedTemporalSelfAttentionDynamicEdgeConvWithoutMask`.'
        b: PairOptTensor = (None, None)
        if isinstance(batch, Tensor):
            b = (batch, batch)
        elif isinstance(batch, tuple):
            assert batch is not None
            b = (batch[0], batch[1])

        edge_index = knn(knn_input, knn_input, self.k, b[0], b[1],
                         num_workers=self.num_workers)
        # edge_index[1] = index_mapper[edge_index[1]]
        return self.propagate(edge_index, x=x, size=None, batch=batch)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))

    def aggregate(self, inputs: Tensor, index: Tensor,
                  batch: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        original_shape = inputs.shape
        # We assume K is fixed and the index tensor is sorted!
        attention_input_shape = list([int(original_shape[0] / self.k)]) + list(original_shape)
        attention_input_shape[1] = self.k
        self_attention_input = inputs.reshape(attention_input_shape)
        attn_output = self.multihead_attn(self_attention_input, self_attention_input, self_attention_input)
        attn_output = attn_output.reshape(original_shape)
        # Apply attention mechanism
        return scatter(attn_output, index, dim=self.node_dim, dim_size=dim_size,
                       reduce=self.aggr)

    def __repr__(self):
        return '{}(nn={}, k={})'.format(self.__class__.__name__, self.nn,
                                        self.k)


class AutomatedGraphDynamicEdgeConv(MessagePassing):
    def __init__(self, nn_before_graph_creation: Union[Callable, None], nn: Callable, graph_creation_in_features: int,
                 in_features: int, head_num: int,
                 k: int, aggr: str = 'max', **kwargs):
        super(AutomatedGraphDynamicEdgeConv,
              self).__init__(aggr=aggr, flow='target_to_source', **kwargs)

        if knn is None:
            raise ImportError('`AutomatedGraphDynamicEdgeConv` requires `torch-cluster`.')
        self.k = k
        self.graph_creator = SelfAttentionEdgeIndexCreatorLayer(graph_creation_in_features, head_num, k)
        self.nn_before_graph_creation = nn_before_graph_creation
        self.nn = nn
        self.multihead_attn = MultiHeadAttention(in_features, head_num)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.multihead_attn)
        reset(self.nn)

    def forward(
            self, x: Union[Tensor, PairTensor],
            sequence_number: Union[Tensor, PairTensor],
            batch: Union[OptTensor, Optional[PairTensor]] = None, ) -> Tensor:
        batch_size = len(np.unique(batch.cpu().numpy()))
        num_point = len(x) // batch_size
        if self.nn_before_graph_creation:
            x = self.nn_before_graph_creation(x)
        graph_creator_input = x.reshape(batch_size, -1, x.shape[-1])
        edge_index = self.graph_creator(graph_creator_input, graph_creator_input)
        point_index_corrector = torch.tensor([i * num_point for i in range(batch_size)]).to(x.device)
        point_index_corrector = point_index_corrector \
            .reshape(-1, 1).repeat(1, num_point * self.k) \
            .reshape(batch_size, num_point * self.k, 1).repeat(1, 1, 2) \
            .permute(0, 2, 1)
        edge_index = (edge_index + point_index_corrector).permute(1, 0, 2).reshape(2, -1)

        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None, batch=batch), edge_index

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))

    def aggregate(self, inputs: Tensor, index: Tensor,
                  batch: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        original_shape = inputs.shape
        # We assume K is fixed and the index tensor is sorted!
        attention_input_shape = list([int(original_shape[0] / self.k)]) + list(original_shape)
        attention_input_shape[1] = self.k
        self_attention_input = inputs.reshape(attention_input_shape)
        attn_output = self.multihead_attn(self_attention_input, self_attention_input, self_attention_input)
        attn_output = attn_output.reshape(original_shape)
        # Apply attention mechanism
        return scatter(attn_output, index, dim=self.node_dim, dim_size=dim_size,
                       reduce=self.aggr)

    def __repr__(self):
        return '{}(nn={}, k={})'.format(self.__class__.__name__, self.nn,
                                        self.k)


class TemporalAutomatedGraphDynamicEdgeConv(MessagePassing):
    def __init__(self, nn_before_graph_creation: Union[Callable, None], nn: Callable, graph_creation_in_features: int,
                 in_features: int, head_num: int,
                 k: int, batch_size, num_points, num_frames, device, aggr: str = 'max', t=5, **kwargs):
        super(TemporalAutomatedGraphDynamicEdgeConv,
              self).__init__(aggr=aggr, flow='target_to_source', **kwargs)
        assert t % 2 == 1, 't should be an odd number'
        if knn is None:
            raise ImportError('`TemporalAutomatedGraphDynamicEdgeConv` requires `torch-cluster`.')
        self.k = k
        self.graph_creator = TemporalSelfAttentionEdgeIndexCreatorLayer(graph_creation_in_features * 2, head_num, k,
                                                                        num_points=num_points, device=device)
        self.nn_before_graph_creation = nn_before_graph_creation
        self.nn_for_seq_num = MLP([1, graph_creation_in_features])
        self.nn = nn
        self.t = t
        self.multihead_attn = MultiHeadAttention(in_features, head_num)
        num_point_per_frame = num_points // num_frames
        self.mask = torch.zeros(num_points, num_points).to(device)
        for frame in range(num_frames):
            start_row_index = (frame - self.t // 2) * num_point_per_frame
            if start_row_index < 0:
                start_row_index = 0
            end_row_index = (frame + self.t // 2) * num_point_per_frame
            start_col_index = frame * num_point_per_frame
            end_col_index = (frame + 1) * num_point_per_frame
            self.mask[start_row_index:end_row_index, start_col_index:end_col_index] = 1
        self.point_index_corrector = torch.tensor([i * num_points for i in range(batch_size)]).to(device)
        self.point_index_corrector = self.point_index_corrector \
            .reshape(-1, 1).repeat(1, num_points * self.k) \
            .reshape(batch_size, num_points * self.k, 1).repeat(1, 1, 2) \
            .permute(0, 2, 1)
        self.batch_size = batch_size
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.multihead_attn)
        reset(self.nn)
        reset(self.nn_for_seq_num)
        reset(self.nn_before_graph_creation)
        reset(self.graph_creator)

    def forward(
            self, x: Union[Tensor, PairTensor],
            sequence_number: Union[Tensor, PairTensor],
            batch: Union[OptTensor, Optional[PairTensor]] = None, ) -> Tensor:
        batch_size = len(np.unique(batch.cpu().numpy()))
        if self.nn_before_graph_creation:
            x = self.nn_before_graph_creation(x)
        transformed_sequence_number = self.nn_for_seq_num(sequence_number.reshape(-1, 1))
        graph_creator_input = torch.cat((x, transformed_sequence_number), 1)
        graph_creator_input = graph_creator_input.reshape(batch_size, -1, graph_creator_input.shape[-1])
        edge_index = self.graph_creator(graph_creator_input, graph_creator_input, self.mask)
        # We're in the last batch of the epoch
        if self.batch_size != batch_size:
            edge_index = (edge_index + self.point_index_corrector[:batch_size, :, :]).permute(1, 0, 2).reshape(2, -1)
        else:
            edge_index = (edge_index + self.point_index_corrector).permute(1, 0, 2).reshape(2, -1)

        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index.type(torch.long), x=x, size=None, batch=batch), edge_index

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))

    def aggregate(self, inputs: Tensor, index: Tensor,
                  batch: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        original_shape = inputs.shape
        # We assume K is fixed and the index tensor is sorted!
        attention_input_shape = list([int(original_shape[0] / self.k)]) + list(original_shape)
        attention_input_shape[1] = self.k
        self_attention_input = inputs.reshape(attention_input_shape)
        attn_output = self.multihead_attn(self_attention_input, self_attention_input, self_attention_input)
        attn_output = attn_output.reshape(original_shape)
        # Apply attention mechanism
        return scatter(attn_output, index, dim=self.node_dim, dim_size=dim_size,
                       reduce=self.aggr)

    def __repr__(self):
        return '{}(nn={}, k={})'.format(self.__class__.__name__, self.nn,
                                        self.k)
