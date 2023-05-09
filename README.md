# DLNLP_assignment_23
## environment build and run project
First install cuda11.6 on the official website (install it yourself according to the compatible version of this machine)
Install torch1.13.1 + cuda11.6 in conda python==3.7 environment (choose according to the installed cuda version)
Run recommended PyCharm + conda environment
## pre-trained BERT model download
You can choose to load when running the code, or you can download it locally in advance. The latter download method is recommended
Download directly from git
git lfs install
git clone https://huggingface.co/bert-base-uncased
Or download all files directly in the folder bert-base-uncased at https://huggingface.co/bert-base-uncased
## Task A model train
The code will save the model at five checkpoints, and the model of checkpoint-2565 is used in the report
After downloading bert-base-uncased and placing it in the specified folder, run taskA.py, and then run loadModel.py
## Task B and C dataset download
nlp_chinese_corpus git address https://github.com/brightmart/nlp_chinese_corpus
Download the json version of the encyclopedia question and answer (baike2018qa) Download the json version of the encyclopedia question and answer (baike2018qa)
After downloading, put it in the baike_qa2019 folder of Task B, which contains two folders, one is the training set and the other is the verification set
If you want to adjust the network structure by yourself, you need to download the training data or replace the data yourself. If you want to directly see the accuracy of the model or use the model directly, run test.py directly
(the model folder already contains a trained model)
## Task D 
First run taskD.py to train the model, and then run loadModel.py to load and use the model. (the model folder already contains a trained model)
## Task E
Adjust the train_test_optim.py comment part of the code to generate different models, run test_train_optim.py to test the accuracy, pay attention to change the folder name, otherwise it will be overwritten.
