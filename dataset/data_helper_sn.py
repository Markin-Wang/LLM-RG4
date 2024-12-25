import gc
import os
import json
import re
import numpy as np
from PIL import Image
import torch.utils.data as data
from transformers import BertTokenizer, AutoImageProcessor
import pandas as pd
import random
from config.config import parser
import math
import torch
from tqdm import tqdm

class FieldParser:
    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        self.rad_dino_processor = AutoImageProcessor.from_pretrained(args.rad_dino_path)


    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values[0] 

    # from https://github.com/cuhksz-nlp/R2Gen/blob/main/modules/tokenizers.py
    def clean_report(self, report):
        # clean Iu-xray reports
        if self.dataset == "iu_xray":
            report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                            replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'
        # clean MIMIC-CXR reports
        else:
            report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
                .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
                .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
                .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
                .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace(':', ' :') \
                .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '')
                                .replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'
        return report

    def parse(self, features):
        to_return = {'id': features['id']}
        to_return['dataset_id'] = 'sn'
        to_return['finding_flag'] = True
        to_return['impression_flag'] = True
        report = ''
        finding = features['finding']
        impression = features['impression']
        if finding != 's' and impression != 's':
            report = finding + ' ' + impression
        else:
            if finding == 's':
                report = impression
                to_return['finding_flag'] = False
            if impression == 's':
                report = finding
                to_return['impression_flag'] = False

        to_return['APPA_flag'] = features['APPA_flag']
        report = self.clean_report(report)
        to_return['input_text'] = report

        # chest x-ray images
        APPA_imagepath = features['APPA_imagepath']
        image = Image.open(self.args.base_dir + APPA_imagepath)
        inputs = self.rad_dino_processor(images=image, return_tensors="pt")
        images = inputs.data['pixel_values']
        to_return["image"] = torch.squeeze(images, 0)  # image tensor
        to_return['last_report'] = 's'
        scores = features['new_scores']
        scores = [float(i) for i in scores]
        scores = torch.tensor(scores, dtype=torch.float32)
        scores = torch.nn.functional.pad(scores, (0, 100 - scores.size(0)), 'constant', 0)
        to_return['scores'] = scores

        history = features['history']
        indication = features['indication']
        h_i = ''
        if indication != 0 or history != 0:
            if indication == 0:
                h_i = 'HISTORY: ' + history
            if history == 0:
                h_i = 'INDICATION: ' + indication
            if history != 0 and indication != 0:
                h_i = 'INDICATION: ' + indication + ' HISTORY: ' + history
        to_return['h_i'] = h_i

        return to_return

    def transform_with_parse(self, inputs):
        return self.parse(inputs)


class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.meta = json.load(open(args.sn_annotation, 'r'))
        self.meta = self.meta[split]
        self.parser = FieldParser(args)
        self.dataset = args.dataset

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.meta[index])

def create_datasets_sn(args):
    train_dataset = ParseDataset(args, 'train')
    dev_dataset = ParseDataset(args, 'val')
    test_dataset = ParseDataset(args, 'test')
    return train_dataset,dev_dataset,test_dataset



