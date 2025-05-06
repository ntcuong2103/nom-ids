
# Image dataset class pytorch
import os
import PIL
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from trie_search import Trie
from typing import List
from torch import FloatTensor, LongTensor
import pytorch_lightning as pl
from torch.utils.data import DataLoader

base_vocab = open('vocab_ids.txt', 'r').read().split('\n')
ids_dict = {line.strip().split('\t')[0]:line.strip().split('\t')[1] for line in open('ids_exp.txt', 'r').readlines()}

class Vocab:
    def __init__(self, base_vocab, ids_dict):
        self.id2char = {i: c for i, c in enumerate(base_vocab)}
        self.char2id = {c: i for i, c in self.id2char.items()}
        self.size = len(base_vocab)
        self.ids_dict = ids_dict
        self.ids_dict_rev = {v: k for k, v in ids_dict.items()}

        self.trie = Trie()
        for k, v in ids_dict.items():
            self.trie.insert(self.encode(k))
        
    def __len__(self):
        return self.size
    
    def encode(self, c):
        return [self.char2id[c] for c in self.ids_dict[c]]

    def decode(self, ids):
        closest = self.trie.search_fuzzy(ids, max_distance=5)
        if len(closest) > 0:
            return self.ids_dict_rev[''.join([self.id2char[i] for i in closest[0][0]])]
        return None

class SeqVocab(Vocab):
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2

    def __init__(self, base_vocab, ids_dict):
        base_vocab = ['<pad>', '<sos>', '<eos>'] + base_vocab
        super().__init__(base_vocab, ids_dict)


class ImageDataset(Dataset):
    def __init__(self, data_dir, vocab, transform=None, num_samples=10):
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize(size=120, max_size=128),
            transforms.RandomInvert(p=1.0),
        ]) 
        self.vocab = vocab
        self.image_paths = []
        self.label_paths = []
        self.num_samples = num_samples
        self.load_data()

    def load_data(self):
        for root, _, files in os.walk(f'{self.data_dir}/images'):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))
                    self.label_paths.append(os.path.join(self.data_dir, 'labels', file.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt')))

                    # check if label file exists
                    if not os.path.exists(self.label_paths[-1]):
                        print(f"Label file not found for {file}. Skipping.")
                        self.image_paths.pop()
                        self.label_paths.pop()

    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        image = Image.open(image_path).convert('RGB')

        # read label YOLO format
        with open(label_path, 'r') as f:
            labels = f.readlines()
        classes = [label.strip().split()[0] for label in labels]
        bboxes = [list(map(float, label.strip().split()[1:])) for label in labels]
        classes = np.array(classes)
        bboxes = np.array(bboxes)
        bboxes = bboxes * np.array([image.width, image.height, image.width, image.height])
        bboxes = bboxes.astype(int)

        # random select num_samples
        num_samples = min(len(bboxes), self.num_samples)
        indices = np.random.choice(len(bboxes), num_samples, replace=False)
        bboxes = bboxes[indices]
        classes = classes[indices]
    
        # extract the images by bboxes
        images = []
        out_classes = []
        for bbox,cls in zip(bboxes, classes):
            if cls not in self.vocab.ids_dict:
                continue
            x, y, w, h = bbox
            x1 = max(0, int(x - w / 2))
            y1 = max(0, int(y - h / 2))
            x2 = min(image.width, int(x + w / 2))
            y2 = min(image.height, int(y + h / 2))
            # crop the image
            image_cropped = image.crop((x1, y1, x2, y2))

            if self.transform:
                image_cropped = self.transform(image_cropped)
            
            width, height = image_cropped.size

            # pad the image to 128x128
            pad_x = max(0, 128 - width)
            pad_y = max(0, 128 - height)
            image_cropped_norm = Image.new('RGB', (128, 128), (0, 0, 0))
            image_cropped_norm.paste(image_cropped, (pad_x // 2, pad_y // 2))

            # os.makedirs(os.path.join(self.data_dir, 'crops'), exist_ok=True)
            # image_cropped_norm.save(os.path.join(self.data_dir, 'crops', f'{idx}_{cls}.png'))
            
            images.append(image_cropped_norm)
            out_classes.append(cls)
        
        return [f'{path}@{idx}' for path, idx in zip([image_path] * len(indices), indices) ], images, [self.vocab.encode(cls) for cls in out_classes]


from dataclasses import dataclass

@dataclass
class Batch:
    img_bases: List[str]  # [b,]
    imgs: FloatTensor  # [b, 3, H, W]
    mask: LongTensor  # [b, H, W]
    indices: List[List[int]]  # [b, l]

    def __len__(self) -> int:
        return len(self.img_bases)

    def to(self, device) -> "Batch":
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.to(device),
            mask=self.mask.to(device),
            indices=self.indices,
        )


def collate_fn(batch):
    fnames, images_x, seqs_y = zip(*batch)
    # stack images
    images_x = [img for imgs in images_x for img in imgs]
    images_x = torch.stack([transforms.ToTensor()(img) for img in images_x], dim=0)
    # mask
    mask_x = torch.ones(images_x.size(0), 128, 128, dtype=torch.long)

    # 
    seqs_y = [seq for seqs in seqs_y for seq in seqs]

    return Batch(fnames, images_x, mask_x, seqs_y)


# Lightning datamodule
class ImageDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, vocab, batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.vocab = vocab
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        dataset = ImageDataset(self.data_dir, self.vocab)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn)

    def val_dataloader(self):
        dataset = ImageDataset(self.data_dir, self.vocab)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)

if __name__ == '__main__':
    data_dir = 'datasets/tkh-mth2k2/MTH1000'  
    
    dataset = ImageDataset(data_dir, SeqVocab(base_vocab, ids_dict), num_samples=10)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    print(batch.img_bases)
    print(batch.imgs.size())
    print(batch.mask.size())
    print(batch.indices)
