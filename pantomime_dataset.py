import h5py
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data


class PantomimeDataset(InMemoryDataset):
    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None):
        super(PantomimeDataset, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['ply_data_train.h5', 'ply_data_test.h5']

    @property
    def processed_file_names(self):
        return ['ply_data_train.pt', 'ply_data_test.pt']

    def download(self):
        pass

    def process(self):
        torch.save(self.process_set(self.raw_paths[0]), self.processed_paths[0])
        torch.save(self.process_set(self.raw_paths[1]), self.processed_paths[1])

    def process_set(self, dataset):
        print(dataset)
        original_data = h5py.File(dataset)
        data = np.array(original_data['data'])
        label = np.array(original_data['label'])

        data_list = []
        for gesture_data, gesture_label in zip(*(data, label)):
            gesture_with_seq_number = []
            for frame_index, frame in enumerate(gesture_data):
                for point in frame:
                    gesture_with_seq_number.append(np.array([frame_index + 1, point[0], point[1], point[2]]))
            gesture = Data()
            gesture.x = torch.tensor(gesture_with_seq_number)
            gesture.y = torch.tensor(gesture_label)
            gesture.pos = gesture.x[:, 1:4]
            data_list.append(gesture)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return self.collate(data_list)

