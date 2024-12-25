
import re
import torch.utils.data as data
import pandas as pd
import torch
from tqdm import tqdm

class ParseDataset(data.Dataset):
    def __init__(self, road, split='train'):
        self.road = road
        df = pd.read_csv(self.road)
        selected_data = df.iloc[:, 0:9]
        self.finding_list = []
        self.impression_list = []
        self.split = split
        self.study_id_list = []
        self.report_list = []
        self.prior_study_list = []
        self.prior_proc_list = []
        self.view_list = []
        self.comm_list = []

        if split == 'train' or split == 'val':
            self.study_id_list = selected_data.iloc[:, 0].tolist()
            self.report_list = selected_data.iloc[:, 1].tolist()
            self.prior_study_list = selected_data.iloc[:, 2].tolist()
            self.prior_proc_list = selected_data.iloc[:, 4].tolist()
            self.view_list = selected_data.iloc[:, 6].tolist()
            self.comm_list = selected_data.iloc[:, 8].tolist()

        if split == 'test':
            self.finding_list = selected_data.iloc[:, 0].tolist()
            self.impression_list = selected_data.iloc[:, 1].tolist()
            self.prior_study_list = selected_data.iloc[:, 2].tolist()
            self.prior_proc_list = selected_data.iloc[:, 3].tolist()
            self.view_list = selected_data.iloc[:, 6].tolist()
            self.comm_list = selected_data.iloc[:, 4].tolist()

        if split == 'pred':
            self.study_id_list = selected_data.iloc[:, 0].tolist()
            self.report_list = selected_data.iloc[:, 1].tolist()

            x = 1

    def __len__(self):
        if self.split != 'pred':
            return len(self.prior_study_list)
        else:
            return len(self.study_id_list)

    def __getitem__(self, index):
        dict = {}
        if self.split != 'pred':
            study_id = ''
            report = ''
            if self.split == 'train' or self.split == 'val':
                study_id = self.study_id_list[index]
                report = self.clean_report(self.report_list[index])
            else:
                finding = str(self.finding_list[index])
                impression = str(self.impression_list[index])
                if finding == '0.0':
                    report = 'impression: '+impression
                    report = self.clean_report(report)
                if impression == '0.0':
                    report = 'finding: '+finding
                    report = self.clean_report(report)
                if finding != '0.0' and impression != '0.0':
                    report = 'finding: ' + finding + 'impression: ' + impression
                    report = self.clean_report(report)
            prior_study = self.prior_study_list[index]
            prior_proc = self.prior_proc_list[index]
            view = self.view_list[index]
            comm = self.comm_list[index]
            dict = {
                'study_id':study_id,
                'report':report,
                'prior_study':torch.tensor([float(prior_study)]),
                'prior_proc':torch.tensor([float(prior_proc)]),
                'view':torch.tensor([float(view)]),
                'comm':torch.tensor([float(comm)])
            }
        else:
            study_id = str(self.study_id_list[index])

            report = str(self.report_list[index])

            dict = {
                'study_id': study_id,
                'report': report
            }
        return dict

    def clean_report(self, report):
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

def create_datasets(args):
    predict_dataset = ParseDataset(args.predictroad, 'pred')
    return predict_dataset

