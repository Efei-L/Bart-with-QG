from DFGN.layers import *
import DFGN.DFGN_config as config

class GraphFusionNet(nn.Module):
    """
    Packing Query Version
    """
    def __init__(self):
        super(GraphFusionNet, self).__init__()
        self.n_layers = config.n_layers
        self.max_query_length = 20

        self.bi_attention = BiAttention(input_dim=config.input_dim,
                                        memory_dim=config.input_dim,
                                        hid_dim=config.hidden_dim,
                                        dropout=config.bi_attn_drop)
        self.bi_attn_linear = nn.Linear(config.hidden_dim * 4, config.hidden_dim)

        h_dim = config.hidden_dim
        q_dim = config.hidden_dim if config.q_update else config.input_dim

        self.basicblocks = nn.ModuleList()
        self.query_update_layers = nn.ModuleList()
        self.query_update_linears = nn.ModuleList()

        for layer in range(self.n_layers):
            self.basicblocks.append(BasicBlock(h_dim, q_dim, layer, config))
            if config.q_update:
                self.query_update_layers.append(BiAttention(h_dim, h_dim, h_dim, config.bi_attn_drop))
                self.query_update_linears.append(nn.Linear(h_dim * 4, h_dim))

        q_dim = h_dim if config.q_update else config.input_dim
        # self.entity_fc = nn.Linear(384, 2)
        self.entity_linear_0 = nn.Linear(h_dim + q_dim, h_dim)
        self.entity_linear_1 = nn.Linear(h_dim, 1)
        print("DFGN..gfn ")
    def forward(self, batch,encoder_hidden_states,sent_vec_node):
        query_mapping = batch['query_mask']
        entity_mask = batch['entity_mask']
        context_encoding = encoder_hidden_states

        # extract query encoding

        trunc_query_mapping = query_mapping[:, :self.max_query_length].contiguous()
        trunc_query_state = (context_encoding * query_mapping.unsqueeze(2))[:, :self.max_query_length, :].contiguous()
        # bert encoding query vec
        query_vec = mean_pooling(trunc_query_state, trunc_query_mapping)
        attn_output, trunc_query_state = self.bi_attention(context_encoding, trunc_query_state, trunc_query_mapping)
        input_state = self.bi_attn_linear(attn_output)
        if config.q_update:
            query_vec = mean_pooling(trunc_query_state, trunc_query_mapping)

        softmasks = []
        entity_state = None
        for l in range(self.n_layers):
            context_state, input_state, entity_state, softmask = self.basicblocks[l](input_state, query_vec, batch,sent_vec_node)
            softmasks.append(softmask)
            if config.q_update:
                query_attn_output, _ = self.query_update_layers[l](trunc_query_state, entity_state, entity_mask)
                trunc_query_state = self.query_update_linears[l](query_attn_output)
                query_vec = mean_pooling(trunc_query_state, trunc_query_mapping)
        trip_entity_state = entity_state[:,1:]
        expand_query = query_vec.unsqueeze(1).repeat((1, trip_entity_state.shape[1], 1))
        entity_logits = self.entity_linear_0(torch.cat([trip_entity_state, expand_query], dim=2))
        entity_logits = self.entity_linear_1(F.relu(entity_logits))
        entity_prediction = entity_logits.squeeze(2) - 1e30 * (1 - entity_mask)
        # logits = self.entity_fc(entity_state)
        input_state = torch.cat([context_state,input_state],dim=1)
        return entity_prediction, input_state, entity_state





