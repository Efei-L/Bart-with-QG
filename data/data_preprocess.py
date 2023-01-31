from tqdm import tqdm
import spacy
import en_core_web_sm
import pickle
nlp = en_core_web_sm.load()
import torch
import json
import flair
from EntityGraph import EntityGraph
def feature_extra():
    with open('data.valid.json', "r", encoding='utf-8') as f:
        raw_data = json.load(f)

    all_example = []
    for data in raw_data:
        title_list = dict()
        evidence = data['evidence']

        for i,e in enumerate(evidence):
            if(title_list.get(e['index'][0])==None):
                title_list[e['index'][0]] = [e['text'][0]]
            else:
                title_list[e['index'][0]].append(e['text'][0])
        all_example.append(title_list)

    context = []
    for data in  all_example:


        para = []
        for title in data:
            sents = []
            for sent in data[title]:
                sents.append(sent)
            para.append([title,sents])
        context.append(para)
    return context
def sentence_pro(sentence):
    context = []
    start_ids = 0
    for idx in range(len(sentence)):
        if(sentence[idx] == '.' or sentence[idx] == '?' or sentence[idx] == '!'):
            context.append(sentence[start_ids:idx+1])
            start_ids = idx + 1
    return context
def get_title():
    with open('data.valid.json', "r", encoding='utf-8') as f:
        raw_data = json.load(f)
    all_titles = []
    for data in raw_data:
        title_list = []
        evidence = data['evidence']
        for i,e in enumerate(evidence):
            title_list.append(e['index'][0].replace(" ",""))
        all_titles.append(title_list)
    return all_titles

if __name__ == '__main__':
    ner_tagger = flair.models.SequenceTagger.load('ner')

    context = feature_extra()
    avg_len = 0
    max_len = 0
    min_len = 9999
    i=0
    zero_e = 0
    entity_graph = EntityGraph(context_length=512,tagger=ner_tagger)
    # context = context[514:]
    ar = list()
    tgt = open('valid.tgt.txt', encoding='utf-8').readlines()
    ans = open('valid.ans.txt', encoding='utf-8').readlines()

    for idx,batch in tqdm(enumerate(context),total=len(context)):
        # ans_s = ans[idx].replace("\n","")
        tgt_i = tgt[idx].replace("\n","")
        ans_i = ans[idx].replace("\n","")
        graph = entity_graph.process_example(ner_tagger,context[idx],tgt_i,ans_i)
        ar.append(graph)
        # i+=1
    all_data = {"myData":ar}
    with open("valid_new.json", 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False)