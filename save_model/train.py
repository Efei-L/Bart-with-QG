import torch.optim as optim
from data_util_new import Dataset, collate_fn
from tqdm import tqdm
from model import Model
import numpy
import random
import config_file
from sklearn.metrics import accuracy_score
import time
import torch

print(torch.__version__)  # 查看torch当前版本号

print(torch.version.cuda)  # 编译当前版本的torch使用的cuda版本号

print(torch.cuda.is_available())
class Trainer(object):
    def __init__(self):
        train_dataset = Dataset('train.json', 'train.tgt.txt', 'train.ans.txt', config_file.debug)
        valid_dataset = Dataset('valid.json', 'valid.tgt.txt', 'valid.ans.txt', config_file.debug)
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
        print("using train_1")
        total = sum([param.nelement() for param in self.model.parameters()])
        print('model param size:', total)
        self.lr = config_file.lr
        # self.optimizer = AdamW(self.model.parameters(), lr = self.lr_rate)
        # params_dict = [{'params': self.model.decode.parameters(), 'lr': 0.0004},
        #                {'params': self.model.bart.parameters(), 'lr': 3e-5}]
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.binary_criterion = torch.nn.BCEWithLogitsLoss(size_average=True)
        self.metric_func = lambda y_pred, y_true: accuracy_score(y_true.data.numpy(), y_pred.data.numpy() > 0.5)
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
        for epoch in range(25):
            print('start epoch:', epoch + 1)
            start_time = time.time()
            self.model.train()
            total_train_loss = 0
            train_count = 0
            total_valid_loss = 0
            valid_count = 0
            total_acc = 0
            entity_loss = 0
            if ((epoch + 1) >= 12 and (epoch + 1) % 4 == 0):
                print("lr decay")
                self.lr *= 0.5
                state_dict = self.optimizer.state_dict()
                for param_group in state_dict["param_groups"]:
                    param_group["lr"] = self.lr
                self.optimizer.load_state_dict(state_dict)
            for i, train_data in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                input, label, input_attention_mask, label_attention_mask, target = train_data['context_ids'], \
                                                                                   train_data['label_ids'], train_data[
                                                                                       "context_mask"], train_data[
                                                                                       'label_mask'], train_data[
                                                                                       'entity_tgt']

                input = input.to(config_file.device)
                input_attention_mask = input_attention_mask.to(config_file.device)
                label_attention_mask = label_attention_mask.to(config_file.device)
                label = label.to(config_file.device)
                decoder_label = label[:, :-1].contiguous()
                label_attention_mask = label_attention_mask[:, :-1].contiguous()
                logits, entity_logits = self.model(input, input_attention_mask, decoder_label, label_attention_mask,
                                                   self.schedule_prob, train_data)
                # compute train loss
                batch_size = logits.size(0)
                seq_len = logits.size(1)
                pred = logits.view(batch_size * seq_len, -1)
                target_label = label[:, 1:].contiguous()
                target_label = target_label.contiguous().view(-1)
                train_loss = self.criterion(pred, target_label)
                total_train_loss += train_loss.item()
                # compute entity loss
                target = target.contiguous().view(-1)
                enetity_pred = entity_logits.contiguous().view(-1)
                entity_loss = self.binary_criterion(enetity_pred, target.float())
                # metric = self.metric_func(enetity_pred.cpu(),target.cpu().float())

                total_loss = train_loss + 0.5 * entity_loss
                train_count += 1
                total_loss = total_loss / accum_step
                total_loss.backward()
                if (i + 1) % accum_step == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            self.model.eval()
            for i, valid_data in tqdm(enumerate(self.valid_loader), total=len(self.valid_loader)):
                with torch.no_grad():
                    input, label, input_attention_mask, label_attention_mask, target = valid_data['context_ids'], \
                                                                                       valid_data['label_ids'], \
                                                                                       valid_data["context_mask"], \
                                                                                       valid_data['label_mask'], \
                                                                                       valid_data['entity_tgt']
                    input = input.to(config_file.device)
                    input_attention_mask = input_attention_mask.to(config_file.device)
                    label_attention_mask = label_attention_mask.to(config_file.device)
                    label = label.to(config_file.device)
                    decoder_label = label[:, :-1].contiguous()
                    label_attention_mask = label_attention_mask[:, :-1].contiguous()
                    logits, entity_logits = self.model(input, input_attention_mask, decoder_label, label_attention_mask,
                                                       self.schedule_prob, valid_data)
                    batch_size = logits.size(0)
                    seq_len = logits.size(1)
                    pred = logits.view(batch_size * seq_len, -1)
                    target_label = label[:, 1:].contiguous()
                    target_label = target_label.contiguous().view(-1)
                    valid_loss = self.criterion(pred, target_label)
                    total_valid_loss += valid_loss.item()
                    # compute entity accuracy
                    target = target.contiguous().view(-1)
                    enetity_pred = entity_logits.contiguous().view(-1)
                    entity_loss = self.binary_criterion(enetity_pred, target.float())
                    metric = self.metric_func(enetity_pred.cpu(), target.cpu().float())
                    entity_loss += entity_loss.item()
                    total_acc += metric.item()
                    valid_count += 1
            # self.schedule_prob = self.schedule_decay(epoch)
            # if(epoch+1 >= 12):
            #     self.BeamSearcher.decode(self.model)
            #     bleu = get_bleu()
            end_time = time.time()
            use_time = end_time - start_time
            avg_train_loss = total_train_loss / train_count
            avg_valid_loss = total_valid_loss / valid_count
            avg_entity_loss = entity_loss / valid_count
            avg_acc = total_acc / valid_count
            msg = "epoch {} train loss: {:.4f}  - valid loss  : {:.4f} - entity loss  : {:.4f} - acc  : {:.4f} -  time: {}".format(
                epoch + 1, avg_train_loss,
                avg_valid_loss, avg_entity_loss, avg_acc, use_time)
            print(msg)
            if (epoch + 1 >= 18):
                file_name = str(round(avg_valid_loss, 6)) + 'epoch:' + str(epoch + 1)
                save_path = 'save_model_new/' + file_name
                torch.save(self.model.state_dict(), save_path)