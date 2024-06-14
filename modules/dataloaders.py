import numpy as np
import torch
import itertools
import random
from torch.utils.data import DataLoader
from torchvision import transforms

from .datasets import IuxrayMultiImageDataset, IuxraySingleImageDataset, MimiccxrSingleImageDataset


def transform_adjust_contrast():
    def _func(img):
        return transforms.functional.adjust_contrast(img, contrast_factor=1.5)
    return _func


class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle, secondary=False):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        if split == 'train_bt':
            self.split = 'train'
            drop_last = False
        else:
            self.split = split
            drop_last = True if split == 'train' else False
        self.secondary = secondary

        if split == 'train' or split == 'train_bt':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomAffine(degrees=10, shear=5),
                transforms.ColorJitter(contrast=0.2, brightness=0.2),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
            self.seq_transform = None
        else:
            if not self.secondary:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transform_adjust_contrast(),
                    transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))])
            self.seq_transform = None

        if self.secondary:
            self.dataset = IuxraySingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform, seq_transform=self.seq_transform, secondary=self.secondary)
        else:
            self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform, seq_transform=self.seq_transform)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers,
            'drop_last': drop_last,
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        image_id_batch, image_batch, report_ids_batch, report_masks_batch, seq_lengths_batch, pseudo_report_batch, pseudo_mask_batch, pseudo_lengths_batch = zip(*data)
        image_batch = torch.stack(image_batch, 0)
        max_seq_length = max(seq_lengths_batch)
        max_seq_length_p = max(pseudo_lengths_batch)

        target_batch = np.zeros((len(report_ids_batch), max_seq_length), dtype=int)
        target_masks_batch = np.zeros((len(report_ids_batch), max_seq_length), dtype=int)
        pseudo_batch = np.zeros((len(pseudo_report_batch), max_seq_length_p), dtype=int)
        pseudo_masks_batch = np.zeros((len(pseudo_report_batch), max_seq_length_p), dtype=int)

        for i, report_ids in enumerate(report_ids_batch):
            target_batch[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(report_masks_batch):
            target_masks_batch[i, :len(report_masks)] = report_masks

        for i, report_ids in enumerate(pseudo_report_batch):
            pseudo_batch[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(pseudo_mask_batch):
            pseudo_masks_batch[i, :len(report_masks)] = report_masks

        return image_id_batch, image_batch, torch.LongTensor(target_batch), torch.FloatTensor(target_masks_batch), torch.LongTensor(pseudo_batch), torch.FloatTensor(pseudo_masks_batch)
