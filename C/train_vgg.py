import torch
import os
import torch.nn as nn
import numpy as np
import time

from model_vgg import textCNN
import sen2inds
import textCNN_data
from vgg import vgg11_bn

word2ind, ind2word = sen2inds.get_worddict('wordLabel.txt')
label_w2n, label_n2w = sen2inds.read_labelFile('label.txt')

textCNN_param = {
    'vocab_size': len(word2ind),
    'embed_dim': 60,
    'class_num': len(label_w2n),
    "kernel_num": 64,
    "kernel_size": [3, 3, 3, 3],
    "dropout": 0.5,
}
dataLoader_param = {
    'batch_size': 128,
    'shuffle': True,
}


def main():
    #init net
    print('init net...')
    net = textCNN(textCNN_param)
    # net = vgg11_bn()
    weightFile = 'weight.pkl'
    # if os.path.exists(weightFile):
    #     print('load weight')
    #     net.load_state_dict(torch.load(weightFile))
    # else:
    #     net.init_weight()
    net.init_weight()
    print(net)

    net.cuda()

    #init dataset
    print('init dataset...')
    dataLoader = textCNN_data.textCNN_dataLoader(dataLoader_param)
    valdata = textCNN_data.get_valdata()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    log = open('log_{}.txt'.format(time.strftime('%y%m%d%H')), 'w')
    log.write('epoch step loss\n')
    log_test = open('log_test_{}.txt'.format(time.strftime('%y%m%d%H')), 'w')
    log_test.write('epoch step test_acc\n')
    print("training...")
    for epoch in range(10):
        for i, (clas, sentences) in enumerate(dataLoader):
            optimizer.zero_grad()
            sentences = sentences.type(torch.LongTensor).cuda()
            clas = clas.type(torch.LongTensor).cuda()
            out = net(sentences)
            loss = criterion(out, clas)
            loss.backward()
            optimizer.step()

            if (i + 1) % 1 == 0:
                print("epoch:", epoch + 1, "step:", i + 1, "loss:", loss.item())
                data = str(epoch + 1) + ' ' + str(i + 1) + ' ' + str(loss.item()) + '\n'
                log.write(data)
        print("save model...")
        torch.save(net.state_dict(), weightFile)
        torch.save(net.state_dict(), "model\{}_model_iter_{}_{}_loss_{:.2f}.pkl".format(time.strftime('%y%m%d%H'), epoch, i, loss.item()))  # current is model_4_layer.pkl
        print("epoch:", epoch + 1, "step:", i + 1, "loss:", loss.item())      


if __name__ == "__main__":
    main()