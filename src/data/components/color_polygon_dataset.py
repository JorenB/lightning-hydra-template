from typing import Tuple

import numpy as np
import torch
from skimage.draw import disk, polygon, polygon_perimeter
from torch.utils.data import Dataset
from torchvision.transforms import GaussianBlur


class ColorPolygonDataset(Dataset):
    def __init__(
        self,
        output_shape: Tuple[int, int],
        num_samples: int,
        polygon_pts: int = 5,
        bg_noise_scale: float = 0.3,
        fg_noise_scale: float = 0.3,
        blur_sigma: float = 2.0,
        *args,
        **kwargs,
    ):
        self.output_shape = tuple(output_shape)
        self.num_samples = num_samples

        self.polygon_pts = polygon_pts
        self.bg_noise_scale = bg_noise_scale
        self.fg_noise_scale = fg_noise_scale
        self.blur_sigma = blur_sigma

    def __getitem__(self, index: int):
        w, h = self.output_shape

        n_pts = int(torch.randint(3, self.polygon_pts + 1, (1,)).item())
        psx = torch.randint(0, w, (n_pts,))
        psy = torch.randint(0, h, (n_pts,))

        feature_rows, feature_columns = polygon(psx, psy)

        pixels = torch.zeros(list(self.output_shape) + [3], dtype=torch.float)
        target = torch.zeros(list(self.output_shape), dtype=torch.long)

        target[feature_rows, feature_columns] = 1

        corners = self.get_bounding_box(target)

        bg_color = torch.randint(0, 2, (3,))
        fg_color = torch.randint(0, 2, (3,))
        disk_color = torch.rand(3)
        if torch.eq(bg_color, fg_color).all():
            fg_color = 1 - fg_color

        bg_color = bg_color.float()
        fg_color = fg_color.float()
        pixels[:, :, :] = bg_color
        pixels[feature_rows, feature_columns, :] = fg_color

        disk_rows, disk_columns = self.make_disk(psx, psy, target)
        pixels[disk_rows, disk_columns, :] = disk_color

        for i in range(3):
            fg_noise = (
                ((-1.0) ** fg_color[i])
                * self.fg_noise_scale
                * torch.rand(self.output_shape, dtype=torch.float)
            )
            bg_noise = (
                ((-1.0) ** bg_color[i])
                * self.bg_noise_scale
                * torch.rand(self.output_shape, dtype=torch.float)
            )

            pixels[:, :, i] += target * fg_noise
            pixels[:, :, i] += (1 - target) * bg_noise

        pixels = torch.clamp(pixels, 0, 1)
        pixels = pixels.permute(2, 0, 1)

        blur = GaussianBlur(kernel_size=9, sigma=self.blur_sigma)
        pixels = blur(pixels)

        return {
            "image": pixels,
            "target_segmentation": target,
            "target_bounding_box": [(corners[0], corners[2]), (corners[1], corners[3])],
            "n_pts": n_pts,
        }

    def make_disk(self, psx, psy, target):
        nzs = torch.nonzero(1 - target)
        rand_idx = torch.randint(0, nzs.shape[0], (1,))[0]

        disk_center = nzs[rand_idx]
        perimeter_rows, perimeter_columns = polygon_perimeter(
            psx.numpy(), psy.numpy(), shape=self.output_shape
        )
        perimeter = torch.stack(
            [torch.tensor(perimeter_rows), torch.tensor(perimeter_columns)], dim=1
        )
        dists = torch.norm((perimeter - disk_center).float(), dim=1)
        radius = torch.randint(4, max(5, int(torch.min(dists))), (1,))[0]
        disk_rows, disk_columns = disk(
            tuple(disk_center.tolist()), radius, shape=self.output_shape
        )
        return disk_rows, disk_columns

    def get_bounding_box(self, target):
        rows, cols = torch.nonzero(target).t()
        x_min = torch.min(cols).item()
        x_max = torch.max(cols).item()
        y_min = torch.min(rows).item()
        y_max = torch.max(rows).item()
        return x_min, x_max, y_min, y_max

    def __len__(self):
        return self.num_samples
