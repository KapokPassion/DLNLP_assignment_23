# # Task A
# from datasets import load_dataset
# from transformers import AutoTokenizer
# from transformers import AutoModelForSequenceClassification
# import numpy as np
#
# from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
# from transformers import EvalPrediction
# import torch
#
# from transformers import TrainingArguments, Trainer

# dataset = load_dataset("sem_eval_2018_task_1", "subtask5.english")
# example = dataset['train'][0]
#
# labels = [label for label in dataset['train'].features.keys() if label not in ['ID', 'Tweet']]
# id2label = {idx:label for idx, label in enumerate(labels)}
# label2id = {label:idx for idx, label in enumerate(labels)}
#
# model = AutoModelForSequenceClassification.from_pretrained("A/model/checkpoint-2565")
# tokenizer = AutoTokenizer.from_pretrained("A/model/checkpoint-2565")
#
# text = "I am not going there."
#
# encoding = tokenizer(text, return_tensors="pt")
# encoding = {k: v.to(model.device) for k,v in encoding.items()}
#
# outputs = model(**encoding)
#
# logits = outputs.logits
#
# sigmoid = torch.nn.Sigmoid()
# probs = sigmoid(logits.squeeze().cpu())
# predictions = np.zeros(probs.shape)
# predictions[np.where(probs >= 0.5)] = 1
# # turn predicted id's into actual label names
# predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
# print(predicted_labels)
#
# # ****************************************************
# def preprocess_data(examples):
#   # take a batch of texts
#   text = examples["Tweet"]
#   # encode them
#   encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
#   # add labels
#   labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
#   # create numpy array of shape (batch_size, num_labels)
#   labels_matrix = np.zeros((len(text), len(labels)))
#   # fill numpy array
#   for idx, label in enumerate(labels):
#     labels_matrix[:, idx] = labels_batch[label]
#
#   encoding["labels"] = labels_matrix.tolist()
#
#   return encoding
#
# encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)
#
# encoded_dataset.set_format("torch")
#
# batch_size = 8
# metric_name = "f1"
#
# args = TrainingArguments(
#     f"model",
#     evaluation_strategy = "epoch",
#     save_strategy = "epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     num_train_epochs=5,
#     weight_decay=0.01,
#     load_best_model_at_end=True,
#     metric_for_best_model=metric_name,
#     #push_to_hub=True,
# )
#
# def multi_label_metrics(predictions, labels, threshold=0.5):
#   # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
#   sigmoid = torch.nn.Sigmoid()
#   probs = sigmoid(torch.Tensor(predictions))
#   # next, use threshold to turn them into integer predictions
#   y_pred = np.zeros(probs.shape)
#   y_pred[np.where(probs >= threshold)] = 1
#   # finally, compute metrics
#   y_true = labels
#   f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
#   roc_auc = roc_auc_score(y_true, y_pred, average='micro')
#   accuracy = accuracy_score(y_true, y_pred)
#   # return as dictionary
#   metrics = {'f1': f1_micro_average,
#              'roc_auc': roc_auc,
#              'accuracy': accuracy}
#   return metrics
#
# def compute_metrics(p: EvalPrediction):
#   preds = p.predictions[0] if isinstance(p.predictions,
#                                          tuple) else p.predictions
#   result = multi_label_metrics(
#     predictions=preds,
#     labels=p.label_ids)
#   return result
#
# trainer = Trainer(
#     model,
#     args,
#     train_dataset=encoded_dataset["train"],
#     eval_dataset=encoded_dataset["validation"],
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )
#
# # print(trainer.evaluate())
# dic = trainer.evaluate()
# for key in dic.keys():
#     print(key, ': ', dic[key])


# ========================================================================================
# # Task B
# import torch
# import os
# import torch.nn as nn
# import numpy as np
# import time
#
# from B.model import textCNN
# import B.sen2inds
#
# word2ind, ind2word = B.sen2inds.get_worddict('B/wordLabel.txt')
# label_w2n, label_n2w = B.sen2inds.read_labelFile('B/label.txt')
#
# textCNN_param = {
#     'vocab_size': len(word2ind),
#     'embed_dim': 60,
#     'class_num': len(label_w2n),
#     "kernel_num": 16,
#     "kernel_size": [3, 4, 5],
#     "dropout": 0.5,
# }
#
#
# def get_valData(file):
#     datas = open(file, 'r').read().split('\n')
#     datas = list(filter(None, datas))
#
#     return datas
#
#
# def parse_net_result(out):
#     score = max(out)
#     label = np.where(out == score)[0][0]
#
#     return label, score
#
#
# # init net
# print('init net...')
# net = textCNN(textCNN_param)
# weightFile = 'B/model/23050512_model_iter_94_195_loss_0.00.pkl'
# if os.path.exists(weightFile):
#     print('load weight')
#     net.load_state_dict(torch.load(weightFile))
# else:
#     print('No weight file!')
#     exit()
# print(net)
#
# net.cuda()
# net.eval()
#
# numAll = 0
# numRight = 0
# testData = get_valData('B/valdata_vec.txt')
# for data in testData:
#     numAll += 1
#     data = data.split(',')
#     label = int(data[0])
#     sentence = np.array([int(x) for x in data[1:21]])
#     sentence = torch.from_numpy(sentence)
#     predict = net(sentence.unsqueeze(0).type(torch.LongTensor).cuda()).cpu().detach().numpy()[0]
#     label_pre, score = parse_net_result(predict)
#     if label_pre == label and score > -100:
#         numRight += 1
#     if numAll % 100 == 0:
#         print('acc:{}({}/{})'.format(numRight / numAll, numRight, numAll))


# ========================================================================================
# # Task C
# import torch
# import os
# import torch.nn as nn
# import numpy as np
# import time
#
# from C.model_vgg import textCNN
# import C.sen2inds
#
# word2ind, ind2word = C.sen2inds.get_worddict('C/wordLabel.txt')
# label_w2n, label_n2w = C.sen2inds.read_labelFile('C/label.txt')
#
# textCNN_param = {
#     'vocab_size': len(word2ind),
#     'embed_dim': 60,
#     'class_num': len(label_w2n),
#     "kernel_num": 64,
#     "kernel_size": [3, 3, 3, 3],
#     "dropout": 0.5,
# }
#
#
# def get_valData(file):
#     datas = open(file, 'r').read().split('\n')
#     datas = list(filter(None, datas))
#
#     return datas
#
#
# def parse_net_result(out):
#     score = max(out)
#     label = np.where(out == score)[0][0]
#
#     return label, score
#
#
# # init net
# print('init net...')
# net = textCNN(textCNN_param)
# weightFile = 'C/model/23050909_model_iter_7_195_loss_0.31.pkl'
# if os.path.exists(weightFile):
#     print('load weight')
#     net.load_state_dict(torch.load(weightFile))
# else:
#     print('No weight file!')
#     exit()
# print(net)
#
# net.cuda()
# net.eval()
#
# numAll = 0
# numRight = 0
# testData = get_valData('C/valdata_vec.txt')
# for data in testData:
#     numAll += 1
#     data = data.split(',')
#     label = int(data[0])
#     sentence = np.array([int(x) for x in data[1:21]])
#     sentence = torch.from_numpy(sentence)
#     predict = net(sentence.unsqueeze(0).type(torch.LongTensor).cuda()).cpu().detach().numpy()[0]
#     label_pre, score = parse_net_result(predict)
#     if label_pre == label and score > -100:
#         numRight += 1
#     if numAll % 100 == 0:
#         print('acc:{}({}/{})'.format(numRight / numAll, numRight, numAll))


# ========================================================================================
# # Task D
# import pandas as pd
# import numpy as np
# from sklearn.metrics import accuracy_score
# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import BertTokenizer, BertConfig, BertForTokenClassification
#
# from torch import cuda
# device = 'cuda' if cuda.is_available() else 'cpu'
# print(device)
#
# model = BertForTokenClassification.from_pretrained('D/model')
# model.to(device)
#
# tokenizer = BertTokenizer.from_pretrained('D/model')
#
# data = pd.read_csv("D/ner_datasetreference.csv", encoding='unicode_escape')
# print(data.head())
#
# print("Number of tags: {}".format(len(data.Tag.unique())))
# frequencies = data.Tag.value_counts()
# print(frequencies)
#
# tags = {}
# for tag, count in zip(frequencies.index, frequencies):
#     if tag != "O":
#         if tag[2:5] not in tags.keys():
#             tags[tag[2:5]] = count
#         else:
#             tags[tag[2:5]] += count
#     continue
#
# print(sorted(tags.items(), key=lambda x: x[1], reverse=True))
# entities_to_remove = ["B-art", "I-art", "B-eve", "I-eve", "B-nat", "I-nat"]
# data = data[~data.Tag.isin(entities_to_remove)]
# print(data.head())
#
# data = data.fillna(method='ffill')
# print(data.head())
#
# data['sentence'] = data[['Sentence #','Word','Tag']].groupby(['Sentence #'])['Word'].transform(lambda x: ' '.join(x))
# # let's also create a new column called "word_labels" which groups the tags by sentence
# data['word_labels'] = data[['Sentence #','Word','Tag']].groupby(['Sentence #'])['Tag'].transform(lambda x: ','.join(x))
# print(data.head())
#
# label2id = {k: v for v, k in enumerate(data.Tag.unique())}
# id2label = {v: k for v, k in enumerate(data.Tag.unique())}
# print(label2id)
#
# data = data[["sentence", "word_labels"]].drop_duplicates().reset_index(drop=True)
# data.head()
#
# MAX_LEN = 128
# TRAIN_BATCH_SIZE = 4
# VALID_BATCH_SIZE = 2
# EPOCHS = 1
# LEARNING_RATE = 1e-05
# MAX_GRAD_NORM = 10
#
# sentence = "India has a capital called Mumbai. On wednesday, the president will give a presentation"
#
# inputs = tokenizer(sentence, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors="pt")
#
# # move to gpu
# ids = inputs["input_ids"].to(device)
# mask = inputs["attention_mask"].to(device)
# # forward pass
# outputs = model(ids, mask)
# logits = outputs[0]
#
# active_logits = logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
# flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size*seq_len,) - predictions at the token level
#
# tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
# token_predictions = [id2label[i] for i in flattened_predictions.cpu().numpy()]
# wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)
#
# word_level_predictions = []
# for pair in wp_preds:
#   if (pair[0].startswith(" ##")) or (pair[0] in ['[CLS]', '[SEP]', '[PAD]']):
#     # skip prediction
#     continue
#   else:
#     word_level_predictions.append(pair[1])
#
# # we join tokens, if they are not special ones
# str_rep = " ".join([t[0] for t in wp_preds if t[0] not in ['[CLS]', '[SEP]', '[PAD]']]).replace(" ##", "")
# print(str_rep)
# print(word_level_predictions)
#
# from transformers import pipeline
#
# pipe = pipeline(task="token-classification", model=model.to("cpu"), tokenizer=tokenizer, aggregation_strategy="simple")
# print(pipe("My name is Niels and New York is a city"))
# my_list = pipe("My name is Niels and New York is a city")
# for i in range(len(my_list)):
#     print(my_list[i])


# ========================================================================================
# # Task E
# import torch
# import os
# import torch.nn as nn
# import numpy as np
# import time
#
# from E.model import textCNN
# import E.sen2inds
#
# word2ind, ind2word = E.sen2inds.get_worddict('E/wordLabel.txt')
# label_w2n, label_n2w = E.sen2inds.read_labelFile('E/label.txt')
#
# textCNN_param = {
#     'vocab_size': len(word2ind),
#     'embed_dim': 60,
#     'class_num': len(label_w2n),
#     "kernel_num": 16,
#     "kernel_size": [3, 4, 5],
#     "dropout": 0.5,
# }
#
#
# def get_valData(file):
#     datas = open(file, 'r').read().split('\n')
#     datas = list(filter(None, datas))
#
#     return datas
#
#
# def parse_net_result(out):
#     score = max(out)
#     label = np.where(out == score)[0][0]
#
#     return label, score
#
#
# # init net
# print('init net...')
# net = textCNN(textCNN_param)
# weightFile = 'B/model/23050512_model_iter_94_195_loss_0.00.pkl'
# if os.path.exists(weightFile):
#     print('load weight')
#     net.load_state_dict(torch.load(weightFile))
# else:
#     print('No weight file!')
#     exit()
# print(net)
#
# net.cuda()
# net.eval()
#
# numAll = 0
# numRight = 0
# testData = get_valData('E/valdata_vec.txt')
# for data in testData:
#     numAll += 1
#     data = data.split(',')
#     label = int(data[0])
#     sentence = np.array([int(x) for x in data[1:21]])
#     sentence = torch.from_numpy(sentence)
#     predict = net(sentence.unsqueeze(0).type(torch.LongTensor).cuda()).cpu().detach().numpy()[0]
#     label_pre, score = parse_net_result(predict)
#     if label_pre == label and score > -100:
#         numRight += 1
#     if numAll % 100 == 0:
#         print('acc:{}({}/{})'.format(numRight / numAll, numRight, numAll))
