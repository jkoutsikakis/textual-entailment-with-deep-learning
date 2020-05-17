import torch
import json

from tqdm.auto import tqdm
from torch.utils.data import Dataset

from .utils.sequences import pad_to_max


class SNLIDataset(Dataset):
    def __init__(self, file, w2i):
        self.ids = []
        self.prems = []
        self.prem_lens = []
        self.hypos = []
        self.hypo_lens = []
        self.targets = []

        for i, l in enumerate(tqdm(file)):
            ex = json.loads(l)
            if ex['label'] == -1:
                continue
            prem, prem_len, hypo, hypo_len = self.process_example(ex, w2i)
            self.prems.append(prem)
            self.prem_lens.append(prem_len)
            self.hypos.append(hypo)
            self.hypo_lens.append(hypo_len)
            self.targets.append(ex['label'])
            self.ids.append(i)

    def __getitem__(self, index):
        return (
            self.ids[index],
            (self.prems[index], self.prem_lens[index], self.hypos[index], self.hypo_lens[index]),
            self.targets[index]
        )

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def collate_fn(batch):
        batch_zipped = list(zip(*batch))
        input_zipped = list(zip(*batch_zipped[1]))

        ids = batch_zipped[0]
        prems = torch.tensor(pad_to_max(input_zipped[0]), dtype=torch.long)
        prem_lens = torch.tensor(input_zipped[1], dtype=torch.int)
        hypos = torch.tensor(pad_to_max(input_zipped[2]), dtype=torch.long)
        hypo_lens = torch.tensor(input_zipped[3], dtype=torch.int)

        target = torch.tensor(batch_zipped[2], dtype=torch.long)

        batch = {
            'id': ids,
            'input': [prems, prem_lens, hypos, hypo_lens],
            'target': target
        }

        return batch

    @staticmethod
    def process_example(ex, w2i):
        premise = [w2i.get(t, 1) for t in ex['premise']]
        hypothesis = [w2i.get(t, 1) for t in ex['hypothesis']]

        return premise, len(premise), hypothesis, len(hypothesis)
