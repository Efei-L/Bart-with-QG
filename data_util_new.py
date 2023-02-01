
import torch
from transformers import BartTokenizer
import config_file
import numpy as np
import json
path ='data/'
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
class Dataset(torch.utils.data.Dataset):
    def __init__(self, src_file, tgt_file, ans_file, debug):
        tgt = open(path + tgt_file, encoding='utf-8').readlines()
        # ans = open(path + ans_file, encoding='utf-8').readlines()
        with open(path + src_file, "r", encoding='utf-8') as f:
            src_raw_data = json.load(f)

        self.src = src_raw_data['myData']
        self.num_seqs = len(self.src)
        if (debug == True):
            if (self.num_seqs > config_file.debug_nums):
                self.src = self.src[:config_file.debug_nums]
                tgt = tgt[:config_file.debug_nums]
                # ans = ans[:config_file.debug_nums]
                self.num_seqs = config_file.debug_nums

        self.tgt = []
        for line in tgt:
            line = line.split('\n')[0]
            self.tgt.append(line)
        # self.ans = []
        # for line in ans:
        #     line = line.split('\n')[0]
        #     self.ans.append(line)




    def __len__(self):
        return self.num_seqs

    def __getitem__(self, i):
        src = self.src[i]
        tgt = self.tgt[i]
        return src, tgt


def collate_fn(data):
    src, tgt = zip(*data)
    max_entity = 0
    max_context_len = 0
    for example in src:
        if example['text_length'] > max_context_len:
            max_context_len = example['text_length']
        if len(example['graph']) > max_entity:
            max_entity = len(example['graph'])
    if(max_entity == 0):
      max_entity = 1

    context_ids = []
    entity_graphs = []
    entity_mapping = []
    entity_mask = []
    entity_lens = []
    query_mask = []
    context_mask = []
    context_length = []
    entity_tgt = []
    for example in src:
        context_ids.append(torch.Tensor(example['tokennums'][:max_context_len]))
        adj = torch.Tensor(example['adj'])
        entity_graphs.append(adj[:max_entity+1,:max_entity+1])
        M = torch.Tensor(example['M'])
        entity_mapping.append(M[:max_entity,:max_context_len])
        entity_nums = len(example['graph'])
        single_entity_mask = torch.zeros(max_entity)
        single_entity_mask[:entity_nums] = 1
        entity_mask.append(single_entity_mask)
        entity_len = []
        for g in example['graph']:
            entity_len.append(len(example['graph'][g]['token_ids']))
        entity_len_c = torch.ones(max_entity)
        entity_len_c[:len(entity_len)] = torch.Tensor(entity_len)
        entity_lens.append(entity_len_c)
        query_m = torch.zeros(max_context_len)
        query_m[:example["ans_len"]] = 1
        query_mask.append(query_m)
        context_m = torch.zeros(512)
        context_m[:example["text_length"]] = 1
        context_mask.append(context_m[:max_context_len])
        context_length.append(example["text_length"])
        extend_entity_y = torch.zeros(max_entity-len(example['entity_y']))
        entity_y_tmp = torch.Tensor(example['entity_y'])
        entity_y = torch.cat((entity_y_tmp, extend_entity_y), dim=-1)
        entity_tgt.append(entity_y)

    context_ids = torch.stack(context_ids).to(torch.int64)
    entity_graphs = torch.stack(entity_graphs).to(torch.float32)
    entity_mapping = torch.stack(entity_mapping).to(torch.float32)
    entity_mask = torch.stack(entity_mask).to(torch.float32)
    entity_lens = torch.stack(entity_lens).to(torch.float32)
    query_mask = torch.stack(query_mask).to(torch.float32)
    context_mask = torch.stack(context_mask).to(torch.float32)
    context_length = torch.Tensor(context_length).to(torch.int64)
    entity_tgt = torch.stack(entity_tgt).to(torch.int64)
    label_data = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=tgt,
        padding='longest',
        return_tensors='pt',
        max_length=32,
        return_attention_mask=True

    )

    label_ids = label_data['input_ids']
    label_mask = label_data['attention_mask']

    return {
        "context_ids":context_ids.to(config_file.device),
        "context_mask":context_mask.to(config_file.device),
        "label_ids":label_ids.to(config_file.device),
        "label_mask":label_mask.to(config_file.device),
        "entity_mapping":entity_mapping.to(config_file.device),
        "entity_mask":entity_mask.to(config_file.device),
        "entity_graphs":entity_graphs.to(config_file.device),
        "entity_lens":entity_lens.to(config_file.device),
        "query_mask":query_mask.to(config_file.device),
        "context_length":context_length.to(config_file.device),
        "entity_tgt": entity_tgt.to(config_file.device)
    }
