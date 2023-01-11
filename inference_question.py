from model import Model
import os
from tqdm import tqdm
from data_util_new import Dataset, collate_fn
import torch
import torch.nn.functional as F
import config_file
import pickle
from transformers import BartTokenizer

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')


class Hypothesis(object):
    def __init__(self, tokens, log_probs, state, context=None):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context

    def extend(self, token, log_prob, state, context=None):
        h = Hypothesis(tokens=self.tokens + [token],
                       log_probs=self.log_probs + [log_prob],
                       state=state,
                       context=context)
        return h

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)


class BeamSearcher(object):
    def __init__(self, model_path, output_dir):
        self.output_dir = output_dir
        self.test_dataset = Dataset('valid.json', 'valid.tgt.txt', 'valid.ans.txt', config_file.debug)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                       batch_size=1,
                                                       collate_fn=collate_fn,
                                                       )
        max_question_len = 32
        self.model = Model(50265)
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)

        self.model.eval()
        self.model = self.model.to(config_file.device)

    @staticmethod
    def sort_hypotheses(hypotheses):
        return sorted(hypotheses, key=lambda h: h.avg_log_prob, reverse=True)

    def decode(self):
        f = open(self.output_dir, encoding='utf-8', mode='w')
        for i, eval_data in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
            input, label, input_attention_mask, label_attention_mask = eval_data['context_ids'], eval_data['label_ids'], \
                                                                       eval_data["context_mask"], eval_data[
                                                                           'label_mask']

            # best_question = self.beam_search(input, label, input_attention_mask, label_attention_mask)
            input = input.to(config_file.device)
            input_attention_mask = input_attention_mask.to(config_file.device)
            sent = self.beam_search(input, input_attention_mask, eval_data)
            sentence = tokenizer.decode(sent.tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            f.writelines(sentence + '\n')
        f.close()

    def beam_search(self, input, input_attention_mask, eval_data):

        prev_context = torch.zeros(1, 1, 2*config_file.hidden_size)
        # prev_context = torch.zeros(1, 1, config_file.hidden_size)
        if config_file.device == 'cuda':
            input = input.to(config_file.device)
            input_attention_mask = input_attention_mask.to(config_file.device)
        # forward encoder

        out = self.model.bart.generate(inputs=input, attention_mask=input_attention_mask, bos_token_id=0,
                                       pad_token_id=1,
                                       eos_token_id=2, decoder_start_token_id=0, max_length=32, min_length=12,
                                       output_hidden_states=True,
                                       num_beams=1,
                                       return_dict_in_generate=True)
        decoder_output = out.sequences
        decoder_output = decoder_output[:, 1:].contiguous()
        one_mat = torch.ones(decoder_output.size(0), decoder_output.size(1)).to(config_file.device)
        zero_mat = torch.zeros(decoder_output.size(0), decoder_output.size(1)).to(config_file.device)
        decoder_mask = zero_mat.eq(one_mat.eq(decoder_output)).to(config_file.device)
        encoder_hidden_state = out.encoder_hidden_states[-1]

        init_hidden_state = encoder_hidden_state[:, 0, :].reshape(config_file.decoder_num_layer, -1, 512).squeeze(1)
        decoder_emb = self.model.embedding_dec(decoder_output)
        decoder_emb = self.model.decoder_trans(decoder_emb)
        encoder_hidden_states = self.model.encoder_trans(encoder_hidden_state)
        init_hidden_state = self.model.init_state_trans(init_hidden_state)
        _,dfgn_hidden_states = self.model.dfgn(eval_data, encoder_hidden_states)
        f1 = self.model.encoder_fc(encoder_hidden_states)
        f2 = self.model.dfgn_fc(dfgn_hidden_states)
        gt = torch.sigmoid(f1 + f2)
        # encoder_features = encoder_hidden_states * gt + dfgn_hidden_states * (1 - gt)
        encoder_features = encoder_hidden_states + dfgn_hidden_states



        hypotheses = [Hypothesis(tokens=[0],
                                 log_probs=[0.0],
                                 state=init_hidden_state,
                                 context=prev_context[0]) for _ in range(config_file.beam_size)]
        # tile enc_outputs, enc_mask for beam search
        encoder_features = encoder_features.repeat(config_file.beam_size, 1, 1)
        input_attention_mask = input_attention_mask.repeat(config_file.beam_size, 1)
        decoder_mask = decoder_mask.repeat(config_file.beam_size, 1)
        decoder_features = decoder_emb.repeat(config_file.beam_size, 1, 1)
        num_steps = 0
        results = []
        while num_steps < config_file.max_decode_step and len(results) < config_file.beam_size:
            latest_tokens = [h.latest_token for h in hypotheses]
            latest_tokens = [idx if idx < 50265 else 2 for idx in latest_tokens]
            prev_y = torch.tensor(latest_tokens, dtype=torch.long).view(-1)

            if config_file.device == 'cuda':
                prev_y = prev_y.to(config_file.device)

            # make batch of which size is beam size
            all_state_h = []
            all_context = []
            for h in hypotheses:
                all_state_h.append(h.state)
                all_context.append(h.context)

            prev_h = torch.stack(all_state_h, dim=1)  # [num_layers, beam, d]
            prev_context = torch.stack(all_context, dim=0)
            prev_context = prev_context.to(config_file.device)
            prev_states = prev_h
            # [beam_size, |V|]
            logits, h_state, context_vector = self.model.decode.decoder(prev_y, prev_states, encoder_features,
                                                                        input_attention_mask, decoder_mask,
                                                                        decoder_features, prev_context)

            log_probs = F.log_softmax(logits, dim=1)
            top_k_log_probs, top_k_ids \
                = torch.topk(log_probs, config_file.beam_size * 2, dim=-1)

            all_hypotheses = []
            num_orig_hypotheses = 1 if num_steps == 0 else len(hypotheses)
            for i in range(num_orig_hypotheses):
                h = hypotheses[i]
                state_i = h_state[:, i, :]
                context_i = context_vector[i]
                for j in range(config_file.beam_size * 2):
                    new_h = h.extend(token=top_k_ids[i][j].item(),
                                     log_prob=top_k_log_probs[i][j].item(),
                                     state=state_i,
                                     context=context_i)
                    all_hypotheses.append(new_h)

            hypotheses = []
            for h in self.sort_hypotheses(all_hypotheses):
                if h.latest_token == 2:
                    if num_steps >= config_file.min_decode_step:
                        results.append(h)
                else:
                    hypotheses.append(h)

                if len(hypotheses) == config_file.beam_size or len(results) == config_file.beam_size:
                    break
            num_steps += 1
        if len(results) == 0:
            results = hypotheses
        h_sorted = self.sort_hypotheses(results)

        return h_sorted[0]