import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BartForConditionalGeneration, BartConfig
from DFGN.GFN import GraphFusionNet
import config_file
epsilon = 1e-20
class Decode(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_size,num_layers):
        super(Decode, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.GRU(embedding_size, hidden_size, batch_first=True,
                            num_layers=num_layers, bidirectional=False)


        # self.context_fc = nn.Linear(3*hidden_size,hidden_size)
        self.contextA_trans = nn.Linear(hidden_size,hidden_size)
        self.contextB_trans = nn.Linear(hidden_size,hidden_size)
        self.output_trans = nn.Linear(hidden_size,hidden_size)
        self.reduce_layer = nn.Linear(embedding_size+hidden_size,embedding_size)
        self.concat_layer = nn.Linear(2*hidden_size,hidden_size)
        self.logit_layer = nn.Linear(hidden_size,vocab_size)

    @staticmethod
    def attention(query, memories, mask):
        # query : [b, 1, d]
        energy = torch.matmul(query, memories.transpose(1, 2))  # [b, 1, t]
        energy = energy.squeeze(1).masked_fill(mask == 0, value=-1e12)
        attn_dist = F.softmax(energy, dim=1).unsqueeze(dim=1)  # [b, 1, t]
        context_vector = torch.matmul(attn_dist, memories)  # [b, 1, d]
        return context_vector, energy
    def sample_pre_y(self, k, ys_e, y_tm_model, ss_prob):
        batch_size = ys_e.size(0)
        if(ss_prob<1):
            y_tm_oracle = y_tm_model
            with torch.no_grad():
                _g = torch.bernoulli(ss_prob * torch.ones(batch_size, 1, requires_grad=False))
                _g = _g.to(config_file.device)
                y_tm = ys_e[:,k,:] * _g + y_tm_oracle * (1. - _g)
        else:
            y_tm = ys_e[:,k,:]
        return y_tm
    def pred_map(self, logit, noise=None):

        if noise is not None:
            with torch.no_grad():
                logit.data.add_( -torch.log(-torch.log(torch.Tensor(
                    logit.size()).cuda().uniform_(0, 1) + epsilon) + epsilon) ) / noise

        return logit

    def forward(self, encoder_outputs, decoder_emb, init_states, encoder_mask, label, decoder_mask,
                scheduled_probs):
        self.lstm.flatten_parameters()
        encoder_memories = encoder_outputs
        decoder_memories = decoder_emb
        batch_size, max_len = label.size()
        hidden_size = encoder_memories.size(-1)
        logits = []

        prev_states = init_states
        prev_context = torch.zeros((batch_size, 1,  hidden_size))
        prev_context = prev_context.to(config_file.device)
        ys_e = self.embedding(label)
        y_tm_model = ys_e[:, 0, :]
        for i in range(max_len):
            embedded = self.sample_pre_y(i, ys_e, y_tm_model, scheduled_probs)
            embedded = embedded.unsqueeze(1)
            lstm_input = self.reduce_layer(
                torch.cat([embedded, prev_context], 2)
            )
            output, states = self.lstm(lstm_input, prev_states)
            contextA, _ = self.attention(output, encoder_memories, encoder_mask)
            contextB, _ = self.attention(output, decoder_memories, decoder_mask)
            contextA_t = self.contextA_trans(contextA)
            contextB_t = self.contextB_trans(contextB)
            output_t = self.output_trans(output)
            g_t = torch.sigmoid(contextA_t + contextB_t + output_t)
            context = g_t * contextA + (1 - g_t) * contextB
            # context = contextA + contextB
            concat_input = torch.cat((output, context), dim=2).squeeze(dim=1)
            logit_input = torch.tanh(self.concat_layer(concat_input))
            logit = self.logit_layer(logit_input)
            logits.append(logit)
            prev_states = states
            # prev_context = torch.cat((contextA, contextB),dim=2)
            prev_context = context
            if scheduled_probs < 1.:
                logit = self.pred_map(logit, 0.1)
                y_tm_model = logit.max(-1)[1]
                y_tm_model = self.embedding(y_tm_model)
        logits = torch.stack(logits, dim=1)
        return logits

    def decoder(self, y, prev_states, encoder_features, encoder_mask, decoder_mask, decoder_features, prev_context ):
        embedded = self.embedding(y.unsqueeze(1))
        lstm_inputs = self.reduce_layer(torch.cat([embedded, prev_context], 2))
        output, states = self.lstm(lstm_inputs, prev_states)
        contextA, _ = self.attention(output, encoder_features, encoder_mask)
        contextB, _ = self.attention(output, decoder_features, decoder_mask)
        # context = torch.cat((contextA,contextB, output),dim=2)
        # g_t = torch.sigmoid(self.context_fc(context))
        contextA_t = self.contextA_trans(contextA)
        contextB_t = self.contextB_trans(contextB)
        output_t = self.output_trans(output)
        g_t = torch.sigmoid(contextA_t + contextB_t + output_t)
        context = g_t * contextA + (1 - g_t) * contextB
        concat_input = torch.cat((output, context), dim=2).squeeze(dim=1)
        logit_input = torch.tanh(self.concat_layer(concat_input))
        logit = self.logit_layer(logit_input)
        # new_context = torch.cat((contextA, contextB), dim=2)
        return logit, states, context

class Model(nn.Module):
    def __init__(self, vocab_size):
        super(Model, self).__init__()
        self.decode = Decode(vocab_size, config_file.hidden_size, config_file.embedding_size,
                             config_file.decoder_num_layer)
        model_path = 'save_model/24.46'
        self.bart = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        state_dict = torch.load(model_path)
        self.bart.load_state_dict(state_dict)
        self.bart = self.bart.to(config_file.device)
        self.embedding_dec = nn.Embedding(vocab_size, config_file.embedding_size)
        self.encoder_trans = nn.Linear(2 * 384, config_file.hidden_size)
        self.decoder_trans = nn.Linear(config_file.embedding_size, config_file.hidden_size)
        self.init_state_trans = nn.Linear(384, config_file.hidden_size)
        self.encoder_fc = nn.Linear(384,384)
        self.sent_node_fc = nn.Linear(768,768)
        self.dfgn_fc = nn.Linear(384,384)
        self.dropout_fc =nn.Linear(3*384,384)
        self.dfgn = GraphFusionNet()
        self.dfgn = self.dfgn.to(config_file.device)
        self.pooler = MultiheadAttPoolLayer(2, 2*config_file.hidden_size, config_file.hidden_size)
        print("恢复原样+gate")
    def get_encoder_features(self, encoder_outputs):
        return self.encoder_trans(encoder_outputs)
    def get_decoder_features(self, decoder_outputs):
        return self.decoder_trans(decoder_outputs)
    def get_init_features(self, init_hidden_state):
        return self.init_state_trans(init_hidden_state)
    def mean_pooling(self,input, mask):
        mean_pooled = input.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        return mean_pooled
    def forward(self, input, input_attention_mask, label, label_attention_mask,scheduled_probs, batch):
        self.bart.eval()
        # self.dfgn.eval()
        with torch.no_grad():

            out = self.bart(input_ids=input, decoder_attention_mask=label_attention_mask,
                            attention_mask=input_attention_mask,
                            decoder_input_ids=label)
            init_hidden_states = out.encoder_last_hidden_state[:, 0, :].reshape(config_file.decoder_num_layer, -1,
                                                                                384)
            encoder_hidden_states = out.encoder_last_hidden_state
            decoder_output = torch.max(out.logits, 2)[1]
            start_tag = torch.zeros(decoder_output.shape[0], dtype=torch.int64).to(config_file.device)
            start_tag = start_tag.unsqueeze(1)
            decoder_output = torch.cat((start_tag, decoder_output), 1)
            decoder_output = decoder_output.to(config_file.device)
            one_mat = torch.ones(decoder_output.size(0), decoder_output.size(1)).to(config_file.device)
            zero_mat = torch.zeros(decoder_output.size(0), decoder_output.size(1)).to(config_file.device)
            decoder_mask = zero_mat.eq(one_mat.eq(decoder_output)).to(config_file.device)
            decoder_mask = decoder_mask.to(config_file.device)
            sent_vec = self.mean_pooling(encoder_hidden_states,input_attention_mask)
            sent_vec = sent_vec.unsqueeze(1)
        sent_vec = self.sent_node_fc(sent_vec)
        decoder_emb = self.embedding_dec(decoder_output)
        decoder_emb = self.decoder_trans(decoder_emb)
        encoder_hidden_states = self.encoder_trans(encoder_hidden_states)
        init_hidden_states = self.init_state_trans(init_hidden_states)
        entity_logits, dfgn_hidden_states, entity_satate = self.dfgn(batch,encoder_hidden_states,sent_vec)
        entity_mask = batch['entity_mask']
        entity_mask = torch.cat((torch.ones((32, 1)).to(config_file.device), entity_mask), dim=1).to(torch.bool)
        graph_vecs, _ = self.pooler(sent_vec.squeeze(1), entity_satate, entity_mask)
        encoder_hidden_states = torch.cat((graph_vecs.unsqueeze(1),encoder_hidden_states),dim=1)
        f1 = self.encoder_fc(encoder_hidden_states)
        f2 = self.dfgn_fc(dfgn_hidden_states)
        gate = torch.sigmoid(f1 + f2)
        encoder_features = encoder_hidden_states * gate + dfgn_hidden_states * (1 - gate)
        input_attention_mask = torch.cat((torch.ones((32,1),dtype=torch.float32).to(input_attention_mask.device),input_attention_mask),dim=1)
        logits = self.decode(encoder_features, decoder_emb, init_hidden_states, input_attention_mask, label,
                            decoder_mask, scheduled_probs)

        return logits, entity_logits


class MultiheadAttPoolLayer(nn.Module):

    def __init__(self, n_head, d_q_original, d_k_original, dropout=0.1):
        super().__init__()
        assert d_k_original % n_head == 0  # make sure the outpute dimension equals to d_k_origin
        self.n_head = n_head
        self.d_k = d_k_original // n_head
        self.d_v = d_k_original // n_head

        self.w_qs = nn.Linear(d_q_original, n_head * self.d_k)
        self.w_ks = nn.Linear(d_k_original, n_head * self.d_k)
        self.w_vs = nn.Linear(d_k_original, n_head * self.d_v)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q_original + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_v)))

        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None):
        """
        q: tensor of shape (b, d_q_original)
        k: tensor of shape (b, l, d_k_original)
        mask: tensor of shape (b, l) (optional, default None)
        returns: tensor of shape (b, n*d_v)
        """
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

        bs, _ = q.size()
        bs, len_k, _ = k.size()

        qs = self.w_qs(q).view(bs, n_head, d_k)  # (b, n, dk)
        ks = self.w_ks(k).view(bs, len_k, n_head, d_k)  # (b, l, n, dk)
        vs = self.w_vs(k).view(bs, len_k, n_head, d_v)  # (b, l, n, dv)

        qs = qs.permute(1, 0, 2).contiguous().view(n_head * bs, d_k)
        ks = ks.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_k)
        vs = vs.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1)
        output, attn = self.attention(qs, ks, vs, mask=mask)

        output = output.view(n_head, bs, d_v)
        output = output.permute(1, 0, 2).contiguous().view(bs, n_head * d_v)  # (b, n*dv)
        output = self.dropout(output)
        return output, attn
class MatrixVectorScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, mask=None):
        """
        q: tensor of shape (n*b, d_k)
        k: tensor of shape (n*b, l, d_k)
        v: tensor of shape (n*b, l, d_v)

        returns: tensor of shape (n*b, d_v), tensor of shape(n*b, l)
        """
        attn = (q.unsqueeze(1) * k).sum(2)  # (n*b, l)
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = (attn.unsqueeze(2) * v).sum(1)
        return output, attn