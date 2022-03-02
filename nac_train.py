# author@Zelo2

from model import nac
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import nac_dataloader
import sklearn.metrics as metrics



def net_save(net, path):
    torch.save(net, path)
    return True


def data_split(dataset_choice):
    if dataset_choice == 'dota2':  # Dota2 Dataset
        data = pd.read_csv('data/dota2018.csv')
    elif dataset_choice == 'lol':  # LOL dataset
        data = pd.read_csv('data/lol75W.csv')

    data = data.values  # Dataframe to Numpy

    outcome = data[:, -1]
    team_information = data[:, :-1]

    match_num = len(outcome)
    hero_num = len(np.unique(team_information))
    team_size = int(len(team_information[0]) / 2)
    print("Dataset:", dataset_choice)
    print("Match Numbers:", match_num)
    print("Hero number in these mathches:", hero_num)
    print("Team size:", team_size)

    '''英雄id并不是按照顺序列出，需要重编码英雄id以防止Embedding出现意外'''
    hero_id = np.unique(team_information)
    hero_dictionary = {}

    for i in range(len(hero_id)):
        hero_dictionary[hero_id[i]] = i

    for i in range(len(team_information)):
        for j in range(len(team_information[0])):
            team_information[i][j] = hero_dictionary[team_information[i][j]]

    data[:, :-1] = team_information  # update hero id

    '''K-fold split'''
    n_split = 5  # 80% train 10% validation 10% test
    kf = KFold(n_splits=n_split, shuffle=True)

    train, validation, test = [], [], []

    '''Split train and (test+validation) data'''
    for i, [train_index, test_plus_validation_index] in enumerate(kf.split(data)):
        train_data = data[train_index]
        test_plus_validation_data = data[test_plus_validation_index]

        '''split test and validation data'''
        kf_tv = KFold(n_splits=2, shuffle=True)
        for test_index, validation_index in kf_tv.split(test_plus_validation_data):
            test_data = test_plus_validation_data[test_index]
            validation_data = test_plus_validation_data[validation_index]

            train.append(train_data)
            validation.append(validation_data)
            test.append(test_data)


    print("*" * 22)
    print("----Data Split End----")


    return [hero_num, team_size], [train, validation, test]



def nac_train(match_information, train_data, vali_data, mark):



    print("----Training Start----")

    hero_num = match_information[0]
    team_size = match_information[1]

    '''Dataset'''
    train_dataset = nac_dataloader.nac_dataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    vali_dataset = nac_dataloader.nac_dataset(vali_data)
    vali_dataloader = DataLoader(vali_dataset, batch_size=32, shuffle=True)

    '''nac net'''
    net = nac.nac_net(embed_dim=20, hero_num=hero_num, team_size=team_size, attention=True)
    net = net.to(device)

    '''loss functional and optimization method'''
    running_loss = 0
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)

    '''Start training'''
    for epoch in range(50):
        print("Epoch", epoch)
        for _, (input_data, label) in enumerate(train_dataloader):

            '''Initialization the gradient'''
            optimizer.zero_grad()

            label = label.float()  # prediction is float type
            label = label.to(device)
            input_data = input_data.to(device)

            prediction = net(input_data)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("Loss:", running_loss)

        '''Validation'''
        vali_preds, vali_labels = []
        for _, (input_data, label) in enumerate(vali_dataloader):
            with torch.no_grad():
                input_data = input_data.to(device)
                prediction = net(input_data)

                prediction = prediction.reshape(-1)
                label = label.reshape(-1)

                vali_preds.append(prediction)
                vali_labels.append(label)

        vali_preds = torch.cat(vali_preds).cpu().numpy()
        vali_labels = torch.cat(vali_labels).cpu().numpy()

        '''AUC'''
        auc = metrics.roc_auc_score(vali_labels, vali_preds)

        '''Acc'''
        vali_preds[vali_preds >= 0.5] = 1
        vali_preds[vali_preds < 0.5] = 0
        acc = np.nanmean((vali_preds == vali_labels) * 1)


        print("Validation of epoch (AUC, Acc)", epoch, ":", auc, acc)


    print("----Training End----")



    net_path = "net_records/nac_net_" + str(mark) + ".pkl"
    torch.save(net.state_dict(), net_path)  # Save the trained nac network

    return net_path


def nac_eval(net_path, match_information, eval_data):
    '''
    :param net_path: The saved nac net
    :param match_information: "hero number" and "team size"
    :param eval_data: dataset for validation
    :return: AUC and Acc(Accuracy)
    '''


    print("----Testing Start----")

    '''Dataset'''
    eval_dataset = nac_dataloader.nac_dataset(eval_data)
    eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=True)


    '''Net work'''
    hero_num = match_information[0]
    team_size = match_information[1]
    net = nac.nac_net(embed_dim=20, hero_num=hero_num, team_size=team_size, attention=True)
    net.load_state_dict(torch.load(net_path))
    net = net.to(device)

    '''Predictions and labels'''
    preds = []
    labels = []

    with torch.no_grad():
        for i, (eval_input, label) in enumerate(eval_dataloader):
            eval_input = eval_input.to(device)
            prediction = net(eval_input)
            # print(prediction)

            prediction = prediction.reshape(-1)
            label = label.reshape(-1)

            preds.append(prediction)
            labels.append(label)

    preds = torch.cat(preds).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()

    '''AUC'''
    # sklearn.metrics.roc_auc_score(y_true, y_score, *, average='macro',
    # sample_weight=None, max_fpr=None, multi_class='raise', labels=None)
    auc = metrics.roc_auc_score(labels, preds)

    '''Acc'''
    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0
    acc = np.nanmean( (preds == labels) * 1)

    print("AUC:", auc, "Acc:", acc)
    print("----Testing End----")
    print("*" * 20)


    return [auc, acc]



if __name__ == '__main__':
    device = torch.device(('cuda:0') if torch.cuda.is_available() else 'cpu')
    dataset_list = ['dota2', 'lol']


    match_information, processed_data = data_split(dataset_choice=dataset_list[0])
    hero_num = match_information[0]
    team_size = match_information[1]

    train_data = processed_data[0]  # [length, true shape of train_dataset]
    vali_data = processed_data[1]
    test_data = processed_data[2]

    auc, acc = [], []

    '''5-fold cross validation'''
    for i in range(len(train_data)):
        net_path = nac_train(match_information, train_data[i], vali_data[i], mark=i+1)
        results_test = nac_eval(net_path, match_information, test_data[i])
        auc.append(results_test[0])
        acc.append(results_test[1])
    print("Performance of NeuralAC:")
    print("AUC:", np.mean(np.array(auc)))
    print("Acc:", np.mean(np.array(acc)))






