import torch.optim as optim
from data_util import Dataset, collate_fn
from tqdm import tqdm
from model_2 import  Model
import numpy
import random
import config_file
from sklearn.metrics import accuracy_score
import time
import torch

print("multi mask bart train ")  # 查看torch当前版本号


print(torch.version.cuda)  # 编译当前版本的torch使用的cuda版本号

print(torch.cuda.is_available())
class Trainer(object):
    def __init__(self):
        train_dataset = Dataset('train.src.txt', 'train.tgt.txt', 'train.ans.txt','train.title.txt', config_file.debug)
        valid_dataset = Dataset('valid.src.txt', 'valid.tgt.txt', 'valid.ans.txt','valid.title.txt', config_file.debug)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=32,
                                                        collate_fn=collate_fn,
                                                        shuffle=True,
                                                        drop_last=True)
        self.valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                        batch_size=32,
                                                        collate_fn=collate_fn,
                                                        shuffle=True,
                                                        drop_last=True)
        print("train size:", len(self.train_loader))
        print("valid size:", len(self.valid_loader))
        self.model = Model(50265)
        print("using train")
        total = sum([param.nelement() for param in self.model.parameters()])
        print('model param size:', total)
        self.lr = config_file.lr
        # self.optimizer = AdamW(self.model.parameters(), lr = self.lr_rate)
        # params_dict = [{'params': self.model.decode.parameters(), 'lr': 0.0004},
        #                {'params': self.model.bart.parameters(), 'lr': 3e-5}]
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        print(torch.cuda.device_count())
        # if (torch.cuda.device_count() > 1):
        #     self.model = nn.DataParallel(self.model, device_ids=[0,1])
        self.model = self.model.to(config_file.device)
        self.schedule_prob = config_file.start_schedule_prob

    def schedule_decay(self, i):
        prob_i = 20 / (20 + numpy.exp((i / 20)))
        return prob_i

    def train(self):
        accum_step = 4
        for epoch in range(10):
            print('start epoch:', epoch + 1)
            start_time = time.time()
            self.model.train()
            total_bart_train_loss = 0
            total_title_train_loss = 0
            train_count = 0
            total_bart_valid_loss = 0
            total_title_valid_loss = 0
            valid_count = 0

            # if ((epoch + 1) >= 20 and (epoch + 1) % 2 == 0):
            #     print("lr decay")
            #     self.lr *= 0.5
            #     state_dict = self.optimizer.state_dict()
            #     for param_group in state_dict["param_groups"]:
            #         param_group["lr"] = self.lr
            #     self.optimizer.load_state_dict(state_dict)
            for i, train_data in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                input_ids, label_ids,input_mask,label_mask,title_ids,title_mask = train_data

                input_ids = input_ids.to(config_file.device)
                input_mask = input_mask.to(config_file.device)
                label_mask = label_mask.to(config_file.device)
                label_ids = label_ids.to(config_file.device)
                title_mask = title_mask.to(config_file.device)
                title_ids = title_ids.to(config_file.device)
                decoder_title_lable = title_ids[:,:-1].contiguous()
                title_mask = title_mask[:,:-1].contiguous()
                decoder_label = label_ids[:, :-1].contiguous()
                label_mask = label_mask[:, :-1].contiguous()
                bart_logits, title_logits = self.model(input_ids, input_mask, decoder_label, label_mask, decoder_title_lable, title_mask,
                                                   self.schedule_prob)
                # compute train loss
                batch_size = bart_logits.size(0)
                seq_len = bart_logits.size(1)
                pred = bart_logits.view(batch_size * seq_len, -1)
                target_label = label_ids[:, 1:].contiguous()
                target_label = target_label.contiguous().view(-1)
                bart_loss = self.criterion(pred, target_label)
                total_bart_train_loss += bart_loss.item()
                # compute entity loss
                batch_size = title_logits.size(0)
                seq_len = title_logits.size(1)
                pred = title_logits.view(batch_size * seq_len, -1)
                target_label = title_ids[:, 1:].contiguous()
                target_label = target_label.contiguous().view(-1)
                title_loss = self.criterion(pred, target_label)
                total_title_train_loss += title_loss.item()
                # metric = self.metric_func(enetity_pred.cpu(),target.cpu().float())

                total_loss = bart_loss + title_loss
                train_count += 1
                total_loss = total_loss / accum_step
                total_loss.backward()
                if (i + 1) % accum_step == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            self.model.eval()
            for i, valid_data in tqdm(enumerate(self.valid_loader), total=len(self.valid_loader)):
                with torch.no_grad():
                    input_ids, label_ids, input_mask, label_mask, title_ids, title_mask = valid_data

                    input_ids = input_ids.to(config_file.device)
                    input_mask = input_mask.to(config_file.device)
                    label_mask = label_mask.to(config_file.device)
                    label_ids = label_ids.to(config_file.device)
                    title_mask = title_mask.to(config_file.device)
                    title_ids = title_ids.to(config_file.device)
                    decoder_title_lable = title_ids[:, :-1].contiguous()
                    title_mask = title_mask[:, :-1].contiguous()
                    decoder_label = label_ids[:, :-1].contiguous()
                    label_mask = label_mask[:, :-1].contiguous()
                    bart_logits, title_logits = self.model(input_ids, input_mask, decoder_label, label_mask,
                                                           decoder_title_lable, title_mask,
                                                           self.schedule_prob)
                    # compute train loss
                    batch_size = bart_logits.size(0)
                    seq_len = bart_logits.size(1)
                    pred = bart_logits.view(batch_size * seq_len, -1)
                    target_label = label_ids[:, 1:].contiguous()
                    target_label = target_label.contiguous().view(-1)
                    bart_loss = self.criterion(pred, target_label)
                    total_bart_valid_loss += bart_loss.item()
                    # compute entity loss
                    batch_size = title_logits.size(0)
                    seq_len = title_logits.size(1)
                    pred = title_logits.view(batch_size * seq_len, -1)
                    target_label = title_ids[:, 1:].contiguous()
                    target_label = target_label.contiguous().view(-1)
                    title_loss = self.criterion(pred, target_label)
                    total_title_valid_loss += title_loss.item()
                    valid_count += 1
            # self.schedule_prob = self.schedule_decay(epoch)
            # if(epoch+1 >= 12):
            #     self.BeamSearcher.decode(self.model)
            #     bleu = get_bleu()
            end_time = time.time()
            use_time = end_time - start_time
            avg_bart_train_loss = total_bart_train_loss / train_count
            avg_title_train_loss = total_title_train_loss / train_count
            avg_bart_valid_loss = total_bart_valid_loss / valid_count
            avg_title_valid_loss = total_title_valid_loss / valid_count
            msg = "epoch {} bart train loss: {:.4f}  -bart valid loss  : {:.4f} - title train loss  : {:.4f} - title valid loss  : {:.4f} -  time: {}".format(
                epoch + 1, avg_bart_train_loss,
                avg_bart_valid_loss, avg_title_train_loss, avg_title_valid_loss, use_time)
            print(msg)
            if (epoch + 1 >= 5):
                file_name = str(round(avg_bart_valid_loss, 6)) + 'epoch:' + str(epoch + 1)
                save_path = 'save_model_new/' + file_name
                torch.save(self.model.state_dict(), save_path)
                print("save success!")


