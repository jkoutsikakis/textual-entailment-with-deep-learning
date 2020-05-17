import torch
import pytorch_wrapper as pw
import pytorch_wrapper.functional as pwF

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LastStateRNNModel(nn.Module):

    def __init__(self,
                 embeddings,
                 projection_size=200,
                 rnn_class=nn.GRU,
                 rnn_bidirectional=True,
                 rnn_depth=2,
                 rnn_size=200,
                 rnn_dp=0.2,
                 mlp_num_layers=1,
                 mlp_hidden_size=200,
                 mlp_activation=nn.ReLU,
                 mlp_dp=0.2):
        super(LastStateRNNModel, self).__init__()

        self.rnn_bidirectional = rnn_bidirectional
        self._embedding_layer = pw.modules.EmbeddingLayer(embeddings.shape[0], embeddings.shape[1], False, 0)
        self._embedding_layer.load_embeddings(embeddings)

        self._linear_projection = nn.Linear(embeddings.shape[1], projection_size, bias=False)

        self._prem_rnn = rnn_class(
            input_size=projection_size,
            hidden_size=rnn_size,
            num_layers=rnn_depth,
            dropout=rnn_dp,
            bidirectional=rnn_bidirectional,
            batch_first=True
        )

        self._hypo_rnn = rnn_class(
            input_size=projection_size,
            hidden_size=rnn_size,
            num_layers=rnn_depth,
            dropout=rnn_dp,
            bidirectional=rnn_bidirectional,
            batch_first=True
        )

        self._out_mlp = pw.modules.MLP(
            input_size=2 * ((projection_size * 2) if rnn_bidirectional else projection_size),
            num_hidden_layers=mlp_num_layers,
            hidden_layer_size=mlp_hidden_size,
            hidden_activation=mlp_activation,
            hidden_dp=mlp_dp,
            output_size=3,
            output_activation=None
        )

    def forward(self, prems_indexes, prem_lens, hypos_indexes, hypo_lens):
        prems = self._embedding_layer(prems_indexes)
        prems = self._linear_projection(prems)
        prems = pack_padded_sequence(prems, prem_lens, batch_first=True, enforce_sorted=False)
        prems = self._prem_rnn(prems)[0]
        prems = pad_packed_sequence(prems, batch_first=True)[0]
        prems_encoding = pwF.get_last_state_of_rnn(prems, prem_lens, self.rnn_bidirectional, is_end_padded=True)

        hypos = self._embedding_layer(hypos_indexes)
        hypos = self._linear_projection(hypos)
        hypos = pack_padded_sequence(hypos, hypo_lens, batch_first=True, enforce_sorted=False)
        hypos = self._hypo_rnn(hypos)[0]
        hypos = pad_packed_sequence(hypos, batch_first=True)[0]
        hypos_encoding = pwF.get_last_state_of_rnn(hypos, hypo_lens, self.rnn_bidirectional, is_end_padded=True)

        encodings = torch.cat([prems_encoding, hypos_encoding], dim=-1)

        return self._out_mlp(encodings)
