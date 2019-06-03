import os

from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

class PennFudanDataset(Dataset):
    def __init__(self, root, transform=None):
        super(PennFudanDataset, self).__init__()

        self.root = root
        self.transforms = transform

        self.imgs = list(sorted(os.listdir(os.path.join(root, 'PNGImages'))))
        self.masks = list(sorted(os.listdir(os.path.join(root, 'PedMasks'))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'PNGImages', self.imgs[idx])
        mask_path = os.path.join(self.root, 'PedMasks', self.masks[idx])

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        num_obj = len(obj_ids)
        boxes = []
        for i in range(num_obj):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_obj,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_obj,), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['imaged_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == '__main__':
    dataset = PennFudanDataset('/Users/miyasatotakaya/Datasets/PennFudanPed/')
    print(dataset.__len__())

    dataloader = DataLoader(dataset, shuffle=False, batch_size=16, num_workers=2, collate_fn=collate_fn)

    for i, (img, target) in enumerate(dataloader):
        print('hoge')