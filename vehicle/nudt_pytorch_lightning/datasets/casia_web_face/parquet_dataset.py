import os
import glob
from typing import List, Optional

import numpy as np
import pandas as pd
from PIL import Image
import io

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class CasiaWebFaceParquet(Dataset):
    def __init__(
        self,
        root_dir: str,
        parquet_glob: str = 'train-*.parquet',
        image_field: str = 'image',
        label_field: str = 'label',
        path_field: Optional[str] = 'path',
        train: bool = True,
        input_shape=(1, 128, 128)
    ):
        super().__init__()
        self.root_dir = root_dir
        self.parquet_glob = parquet_glob
        self.image_field = image_field
        self.label_field = label_field
        self.path_field = path_field
        self.train = train
        self.input_shape = input_shape

        self.transforms = self._build_transforms(input_shape, train)

        parquet_files: List[str] = sorted(glob.glob(os.path.join(root_dir, parquet_glob)))
        if len(parquet_files) == 0:
            raise FileNotFoundError(f'No parquet files matched: {os.path.join(root_dir, parquet_glob)}')

        # Load metadata lazily: only gather offsets (file index, row index)
        self._rows = []
        self._dfs = []
        for fpath in parquet_files:
            df = pd.read_parquet(fpath)
            self._dfs.append(df)
            for idx in range(len(df)):
                self._rows.append((len(self._dfs) - 1, idx))

    @staticmethod
    def _build_transforms(input_shape, train):
        channels, height, width = input_shape
        normalize = T.Normalize(mean=[0.5] * channels, std=[0.5] * channels)
        if train:
            return T.Compose([
                T.Resize((height, width)),
                T.RandomCrop((height, width)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize,
            ])
        else:
            return T.Compose([
                T.Resize((height, width)),
                T.CenterCrop((height, width)),
                T.ToTensor(),
                normalize,
            ])

    def __len__(self):
        return len(self._rows)

    def _load_image(self, row):
        # Prefer raw bytes field; fallback to path
        blob = row[self.image_field] if self.image_field in row else None
        img = None
        if isinstance(blob, (bytes, bytearray)):
            img = Image.open(io.BytesIO(blob))
        elif isinstance(blob, dict):
            data = blob.get('bytes', None)
            if data is not None:
                img = Image.open(io.BytesIO(data))
            elif self.path_field and blob.get('path'):
                img_path = blob['path']
                if not os.path.isabs(img_path):
                    img_path = os.path.join(self.root_dir, img_path)
                img = Image.open(img_path)
        if img is None:
            if self.path_field is None or self.path_field not in row or row[self.path_field] is None:
                raise KeyError('Neither image bytes nor image path available in parquet row')
            img_path = row[self.path_field]
            if not os.path.isabs(img_path):
                img_path = os.path.join(self.root_dir, img_path)
            img = Image.open(img_path)
        return img

    def __getitem__(self, index):
        df_idx, row_idx = self._rows[index]
        row = self._dfs[df_idx].iloc[row_idx]

        label = int(row[self.label_field])
        img = self._load_image(row)
        # convert to grayscale if needed
        if self.input_shape[0] == 1:
            img = img.convert('L')
        else:
            img = img.convert('RGB')

        tensor = self.transforms(img)
        return tensor.float(), label


