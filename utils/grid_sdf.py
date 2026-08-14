import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class TrilinearInterpolation(nn.Module):
    def __init__(self):
        super(TrilinearInterpolation, self).__init__()

    def sample_at_integer_locs(self, input_feats, index_tensor):
        batch_size, num_chans, num_d, height, width = input_feats.shape
        xy_grid = index_tensor[..., 0:2]
        xy_grid[..., 0] = xy_grid[..., 0] - ((width - 1.0) / 2.0)
        xy_grid[..., 0] /= ((width - 1.0) / 2.0)
        xy_grid[..., 1] = xy_grid[..., 1] - ((height - 1.0) / 2.0)
        xy_grid[..., 1] /= ((height - 1.0) / 2.0)
        xy_grid = torch.clamp(xy_grid, min=-1.0, max=1.0)

        sampled_in_2d = F.grid_sample(
            input=input_feats.view(batch_size, num_chans * num_d, height, width),
            grid=xy_grid, mode='nearest', align_corners=True
        ).view(batch_size, num_chans, num_d, xy_grid.shape[1], xy_grid.shape[2])

        z_grid = index_tensor[..., 2].view(batch_size, 1, 1, xy_grid.shape[1], xy_grid.shape[2])
        z_grid = z_grid.long().clamp(min=0, max=num_d - 1)
        z_grid = z_grid.expand(batch_size, num_chans, 1, xy_grid.shape[1], xy_grid.shape[2])
        sampled_in_3d = sampled_in_2d.gather(2, z_grid).squeeze(2)
        return sampled_in_3d

    def forward(self, input_feats, sampling_grid):
        batch_size, num_chans, num_d, height, width = input_feats.shape
        grid_height, grid_width = sampling_grid.shape[1], sampling_grid.shape[2]
        sampling_grid = torch.clamp(sampling_grid, min=-1.0, max=1.0)
        sampling_grid = (sampling_grid + 1) / 2.0
        scaling_factor = torch.FloatTensor([width - 1.0, height - 1.0, num_d - 1.0]).to(input_feats.device).view(1, 1, 1, 3)
        sampling_grid = scaling_factor * sampling_grid
        
        x, y, z = torch.split(sampling_grid, 1, dim=3)
        x_0, y_0, z_0 = torch.floor(x), torch.floor(y), torch.floor(z)
        x_1, y_1, z_1 = x_0 + 1, y_0 + 1, z_0 + 1
        u, v, w = x - x_0, y - y_0, z - z_0
        u, v, w = [t.view(batch_size, 1, grid_height, grid_width).expand(batch_size, num_chans, grid_height, grid_width)
                   for t in (u, v, w)]
        
        c_000 = self.sample_at_integer_locs(input_feats, torch.cat([x_0, y_0, z_0], dim=3))
        c_001 = self.sample_at_integer_locs(input_feats, torch.cat([x_0, y_0, z_1], dim=3))
        c_010 = self.sample_at_integer_locs(input_feats, torch.cat([x_0, y_1, z_0], dim=3))
        c_011 = self.sample_at_integer_locs(input_feats, torch.cat([x_0, y_1, z_1], dim=3))
        c_100 = self.sample_at_integer_locs(input_feats, torch.cat([x_1, y_0, z_0], dim=3))
        c_101 = self.sample_at_integer_locs(input_feats, torch.cat([x_1, y_0, z_1], dim=3))
        c_110 = self.sample_at_integer_locs(input_feats, torch.cat([x_1, y_1, z_0], dim=3))
        c_111 = self.sample_at_integer_locs(input_feats, torch.cat([x_1, y_1, z_1], dim=3))
        
        c_xyz = (1 - u) * (1 - v) * (1 - w) * c_000 + \
                (1 - u) * (1 - v) * w * c_001 + \
                (1 - u) * v * (1 - w) * c_010 + \
                (1 - u) * v * w * c_011 + \
                u * (1 - v) * (1 - w) * c_100 + \
                u * (1 - v) * w * c_101 + \
                u * v * (1 - w) * c_110 + \
                u * v * w * c_111
        return c_xyz