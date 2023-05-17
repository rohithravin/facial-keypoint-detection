import torch
from torch.utils.data import Dataset

class FacialKeypointDataset(Dataset):
  def __init__(self, dataset, transform_list=None):

    [data_X, data_y] = dataset
    X_tensor, y_tensor = data_X, data_y
    tensors = (X_tensor, y_tensor)

    assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)

    self.tensors = tensors
    self.transforms = transform_list

  def __getitem__(self, index):
    x = self.tensors[0][index]
    x = x.permute(2,0,1)

    if self.transforms:
      x = self.transforms(x)

    y = self.tensors[1][index]

    return x, y

  def __len__(self):
    return self.tensors[0].size(0)
  
class FacialKeypointDatasetTest(Dataset):
  def __init__(self, data, transform_list=None):

    self.tensors = data
    self.transforms = transform_list

  def __getitem__(self, index):
    x = self.tensors[index]

    x = x.permute(2,0,1)

    if self.transforms:
      x = self.transforms(x)

    return x

  def __len__(self):
    return self.tensors[0].size(0)