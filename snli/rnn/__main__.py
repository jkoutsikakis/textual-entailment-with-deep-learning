import click
import pickle


@click.group()
def rnn():
    pass


@rnn.command()
def run():
    from ..system_wrapper import SystemWrapper
    from ..dataset import SNLIDataset
    from .config import RNNConfig
    from .model import LastStateRNNModel

    with open('data/decomposed_embeddings.pkl', 'rb') as fr:
        embeddings, w2i, _ = pickle.load(fr)

    model = LastStateRNNModel(embeddings, **RNNConfig['model_params'])
    sw = SystemWrapper(model)

    with open('data/train.jsonl') as fr:
        train_dataset = SNLIDataset(fr, w2i)
    with open('data/validation.jsonl') as fr:
        val_dataset = SNLIDataset(fr, w2i)
    with open('data/test.jsonl') as fr:
        test_dataset = SNLIDataset(fr, w2i)

    sw.train(train_dataset, val_dataset, RNNConfig['training_params'])
    results = sw.evaluate(test_dataset, RNNConfig['training_params']['batch_size'])

    click.echo(results)
