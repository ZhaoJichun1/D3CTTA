import MinkowskiEngine as ME
import torch
import torch.nn as nn
import numpy as np


class DistanceBasedBatchNorm(nn.Module):

  def __init__(self, num_features, dist_range, num_areas, origin_bn):
    super(DistanceBasedBatchNorm, self).__init__()
    self.bns = nn.ModuleList([ME.MinkowskiBatchNorm(num_features).cuda() for _ in range(num_areas)])
    self.num_areas= num_areas
    self.dist_range = dist_range
    self.dist_thre =  np.linspace(dist_range[0], dist_range[1], self.num_areas + 1)

    if origin_bn is not None:
      for bn in self.bns:
          bn.load_state_dict(origin_bn.state_dict())
    # input()



  def forward(self, x):

    coordinates = x.C.float()
    
    # 计算每个点到原点的距离
    distances = torch.sqrt(coordinates[:, 0]**2 + coordinates[:, 1]**2)
    distances = torch.clamp(distances, self.dist_range[0]+1e-3, self.dist_range[1]-1e-3)


    # 根据距离创建掩码，划分三个不同的区域
    x_features = x.F
    x_out_feat = torch.zeros_like(x_features).cuda()
    for i in range(self.num_areas):
      mask = (distances > self.dist_thre[i]) & (distances < self.dist_thre[i+1])
      if mask.any():
        x_feat = x_features[mask]
        x_out_feat[mask] = self.bns[i](ME.SparseTensor(features=x_feat, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)).F

    x_out = ME.SparseTensor(features=x_out_feat, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)
    return x_out


def get_norm(norm_type, num_feats, bn_momentum=0.05, D=-1):
  if norm_type == 'BN':
    return ME.MinkowskiBatchNorm(num_feats, momentum=bn_momentum)
  elif norm_type == 'IN':
    return ME.MinkowskiInstanceNorm(num_feats, dimension=D)
  else:
    raise ValueError(f'Type {norm_type}, not defined')

  