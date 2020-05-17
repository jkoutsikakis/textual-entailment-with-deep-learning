import pytorch_wrapper as pw
import torch
import os
import uuid

from torch import nn
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.optim import Adam


class SystemWrapper:

    def __init__(self, model):

        if torch.cuda.is_available():
            self._system = pw.System(model, last_activation=nn.Softmax(dim=-1), device=torch.device('cuda'))
        else:
            self._system = pw.System(model, last_activation=nn.Softmax(dim=-1), device=torch.device('cpu'))

    def train(self, train_dataset, val_dataset, training_params):

        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=training_params['batch_size'],
            collate_fn=train_dataset.collate_fn
        )

        val_dataloader = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            batch_size=training_params['batch_size'],
            collate_fn=val_dataset.collate_fn
        )

        loss_wrapper = pw.loss_wrappers.GenericPointWiseLossWrapper(nn.CrossEntropyLoss())
        optimizer = Adam(self._system.model.parameters(), lr=training_params['lr'])

        base_es_path = f'/tmp/{uuid.uuid4().hex[:30]}/'
        os.makedirs(base_es_path, exist_ok=True)

        _ = self._system.train(
            loss_wrapper,
            optimizer,
            train_data_loader=train_dataloader,
            evaluation_data_loaders={'val': val_dataloader},
            evaluators={'macro-f1': pw.evaluators.MultiClassF1Evaluator(average='macro')},
            gradient_accumulation_steps=1,
            callbacks=[
                pw.training_callbacks.EarlyStoppingCriterionCallback(
                    patience=10,
                    evaluation_data_loader_key='val',
                    evaluator_key='macro-f1',
                    tmp_best_state_filepath=f'{base_es_path}/temp.es.weights'
                ),
            ]
        )

    def evaluate(self, eval_dataset, batch_size):
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=SequentialSampler(eval_dataset),
            batch_size=batch_size,
            collate_fn=eval_dataset.collate_fn
        )

        evaluators = {

            'acc': pw.evaluators.MultiClassAccuracyEvaluator(),
            'macro-prec': pw.evaluators.MultiClassPrecisionEvaluator(average='macro'),
            'macro-rec': pw.evaluators.MultiClassRecallEvaluator(average='macro'),
            'macro-f1': pw.evaluators.MultiClassF1Evaluator(average='macro')
        }

        return self._system.evaluate(eval_dataloader, evaluators)
