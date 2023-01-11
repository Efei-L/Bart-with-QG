from transformers import BertModel,BartModel
import torch
import torch.nn as nn
from torch.nn.utils import rnn
from torch.autograd import Variable
import torch.nn.functional as F

class MeanMaxPooling(nn.Module):

    def __init__(self):
        super(MeanMaxPooling, self).__init__()
        self.linear = nn.Linear(2*768,768)
    def forward(self, doc_state, entity_mapping, entity_lens):
        """
        :param doc_state:  N x L x d
        :param entity_mapping:  N x E x L
        :param entity_lens:  N x E
        :return: N x E x 2d
        """
        entity_states = entity_mapping.unsqueeze(3) * doc_state.unsqueeze(1)  # N x E x L x d
        max_pooled = torch.max(entity_states, dim=2)[0]
        mean_pooled = torch.sum(entity_states, dim=2) / entity_lens.unsqueeze(2)
        output = torch.cat([max_pooled, mean_pooled], dim=2)
        output = self.linear(output)  # N x E x 2d
        return output
class LSTMWrapper(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layer, concat=False, bidir=True, dropout=0.3, return_last=True):
        super(LSTMWrapper, self).__init__()
        self.rnns = nn.ModuleList()
        for i in range(n_layer):
            if i == 0:
                input_dim_ = input_dim
                output_dim_ = hidden_dim
            else:
                input_dim_ = hidden_dim if not bidir else hidden_dim * 2
                output_dim_ = hidden_dim
            self.rnns.append(nn.LSTM(input_dim_, output_dim_, 1, bidirectional=bidir, batch_first=True))
        self.dropout = dropout
        self.concat = concat
        self.n_layer = n_layer
        self.return_last = return_last

    def forward(self, encoder_hidden_states, entity_state, entity_mapping, input_lengths=None):
        # input_length must be in decreasing order
        expand_entity_state = torch.sum(entity_state.unsqueeze(2) * entity_mapping.unsqueeze(3), dim=1)  # N x E x L x d
        input = torch.cat([expand_entity_state, encoder_hidden_states], dim=2)
        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []

        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()

        for i in range(self.n_layer):
            output = F.dropout(output, p=self.dropout, training=self.training)

            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True,enforce_sorted=False)

            output, _ = self.rnns[i](output)

            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen:  # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)

            outputs.append(output)
        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]
class Bert_Graph_Layer(nn.Module):
    def __init__(self):
        super(Bert_Graph_Layer,self).__init__()
        self.max_mean_pooling = MeanMaxPooling()
        self.ent2doc = LSTMWrapper(768*2, 768 // 2, 1)
        self.bert = BartModel.from_pretrained('facebook/bart-base')
        self.entity_linear_1 = nn.Linear(768, 1)
    def forward(self, batch, encoder_hidden_states):
        entity_mapping = batch['entity_mapping']
        entity_length = batch['entity_lens']
        entity_mask = batch['entity_mask']
        doc_length = batch['context_length']
        src = self.max_mean_pooling(encoder_hidden_states, entity_mapping, entity_length)
        src_mask = entity_mask
        out = self.bert(inputs_embeds = src, attention_mask = src_mask,decoder_inputs_embeds = src, decoder_attention_mask = src_mask)
        d_output = out.last_hidden_state
        doc_state = self.ent2doc(encoder_hidden_states, d_output, entity_mapping, doc_length)
        entity_logits = self.entity_linear_1(d_output)
        return entity_logits, doc_state