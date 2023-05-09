from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import numpy as np

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch

from transformers import TrainingArguments, Trainer

dataset = load_dataset("sem_eval_2018_task_1", "subtask5.english")
example = dataset['train'][0]

labels = [label for label in dataset['train'].features.keys() if label not in ['ID', 'Tweet']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}

model = AutoModelForSequenceClassification.from_pretrained("../A/model/checkpoint-2565")
tokenizer = AutoTokenizer.from_pretrained("../A/model/checkpoint-2565")

text = "I am not going there."

encoding = tokenizer(text, return_tensors="pt")
encoding = {k: v.to(model.device) for k,v in encoding.items()}

outputs = model(**encoding)

logits = outputs.logits

sigmoid = torch.nn.Sigmoid()
probs = sigmoid(logits.squeeze().cpu())
predictions = np.zeros(probs.shape)
predictions[np.where(probs >= 0.5)] = 1
# turn predicted id's into actual label names
predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
print(predicted_labels)

# ****************************************************
def preprocess_data(examples):
  # take a batch of texts
  text = examples["Tweet"]
  # encode them
  encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
  # add labels
  labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
  # create numpy array of shape (batch_size, num_labels)
  labels_matrix = np.zeros((len(text), len(labels)))
  # fill numpy array
  for idx, label in enumerate(labels):
    labels_matrix[:, idx] = labels_batch[label]

  encoding["labels"] = labels_matrix.tolist()

  return encoding

encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)

encoded_dataset.set_format("torch")

batch_size = 8
metric_name = "f1"

args = TrainingArguments(
    f"model",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    #push_to_hub=True,
)

def multi_label_metrics(predictions, labels, threshold=0.5):
  # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
  sigmoid = torch.nn.Sigmoid()
  probs = sigmoid(torch.Tensor(predictions))
  # next, use threshold to turn them into integer predictions
  y_pred = np.zeros(probs.shape)
  y_pred[np.where(probs >= threshold)] = 1
  # finally, compute metrics
  y_true = labels
  f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
  roc_auc = roc_auc_score(y_true, y_pred, average='micro')
  accuracy = accuracy_score(y_true, y_pred)
  # return as dictionary
  metrics = {'f1': f1_micro_average,
             'roc_auc': roc_auc,
             'accuracy': accuracy}
  return metrics

def compute_metrics(p: EvalPrediction):
  preds = p.predictions[0] if isinstance(p.predictions,
                                         tuple) else p.predictions
  result = multi_label_metrics(
    predictions=preds,
    labels=p.label_ids)
  return result

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print(trainer.evaluate())
dic = trainer.evaluate()
for key in dic.keys():
    print(key, ': ', dic[key])
