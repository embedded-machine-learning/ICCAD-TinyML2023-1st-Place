from datetime import datetime
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim.swa_utils import AveragedModel, SWALR
import time

torch.manual_seed(1337)
torch.backends.cudnn.benchmark = True

sys.path.append('./')

from utils import IEGM_DataSET, soft_f1_loss, FlipPeak,Augmentations, true_labels_idx, validation_Fscore

from custom_qat_model import Gatech_net_QAT

def main(count = 0, verbose=True):
    save_name = spezial + str(count)
    # Hyperparameters
    BATCH_SIZE = args.batchsz
    BATCH_SIZE_TEST = args.batchsz
    LR = args.lr
    EPOCH = args.epoch
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices
    weight_decay = args.weight_decay
    result_save_path = args.result_save_path  #Path for storing results and best test model
    loss_function = args.loss_function
    fscore_beta =  args.fscore_beta


    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)
    
    if not os.path.exists('./log/'):
        os.makedirs('./log/')
    t = time.strftime('%Y-%m-%d_%H-%M-%S')
    
    ofp = open(file='log/res_{}.txt'.format(t), mode='w')

    # Instantiating NN
    net = Gatech_net_QAT() 

    l_list_alw = [8]
    l_list_deac = [1,5]
   
    net.train()
    net = net.float().to(device)
    print(device)
    # Start dataset loading
    trainset = IEGM_DataSET(root_dir=path_data,
                            indice_dir=path_indices,
                            # mode=f'train_no_svt',
                            mode=f'train_no_svt',
                            size=SIZE,
                            device='cuda',
                            cache=True,
                            )
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testset = IEGM_DataSET(root_dir=path_data,
                           indice_dir=path_indices,
                           mode=f'test',  
                           size=SIZE,
                           device='cuda',
                           cache=True,
                           )
    testloader = DataLoader(testset, batch_size=BATCH_SIZE_TEST, shuffle=True)

    # augmentations = Augmentations([FlipPeak(),FlipSeg(),AddNoise()])
    # augmentations = Augmentations([FlipPeak(),AddNoise()])
    # augmentations = Augmentations([FlipPeak(),FlipSeg()])
    augmentations = Augmentations([FlipPeak()])
    # augmentations = Augmentations()

    print("Training Dataset loading finish.")

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.cross(from_logits=True)
    
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.005,  weight_decay=weight_decay)
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.01,  weight_decay=weight_decay)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01,momentum=.9,  weight_decay=weight_decay)
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.0005)

    # scheduler = CosineAnnealingLR(optimizer, T_max=100)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10,gamma=.5)
    swa_start = 50
    swa_scheduler = SWALR(optimizer, swa_lr=1e-6, anneal_strategy='linear')

    # criterion = nn.CrossEntropyLoss()
    # # criterion = nn.cross(from_logits=True)
    
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.0005,  weight_decay=weight_decay)
    # #optimizer = torch.optim.SGD(net.parameters(), lr=0.0005)

    # swa_model = AveragedModel(net)
    # scheduler = CosineAnnealingLR(optimizer, T_max=15)
    # swa_start = 10
    # swa_scheduler = SWALR(optimizer, swa_lr=0.0001, anneal_strategy='linear')
    
    #optimizer = torch.optim.Adam(net.parameters(), lr=LR,  weight_decay=weight_decay)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=2e-4) #, eta_max=4e-4

    epoch_num = EPOCH
    Train_loss = []
    Train_acc = []
    Train_precison = []
    Train_recall = []
    Train_fscore = []
    Learning_rate = []
    Test_loss = []
    Test_acc = []
    Test_precison = []
    Test_recall = []
    Test_fscore = []
    Test_G_score = []
    Train_G_score = []
    Test_Full_score = []
    Train_Full_score = []
    
    epoch_list = []
    best_test_fscore = 0.0
    best_test_full_score = 0.0

    print("Start training")
    for epoch in range(epoch_num):  # loop over the dataset multiple times (specify the #epoch)
        running_loss = 0.0
        correct = 0.0
        total = 0.0
        precision = 0.0
        recall = 0.0
        fscore = 0.0
        i = 0
        accuracy = 0.0
        true_Lables = np.zeros((len(true_labels_idx)))
        
        if (epoch-swa_start)%25 == 0 and epoch >= swa_start:
            v = .5**((epoch-swa_start)//25)
            optimizer = torch.optim.SGD(net.parameters(), lr=LR,momentum=.9,  weight_decay=weight_decay)
            # optimizer = torch.optim.Adam(net.parameters(), lr=LR,  weight_decay=weight_decay)
            swa_scheduler = SWALR(optimizer, swa_lr=LR/10,anneal_epochs=20, anneal_strategy='linear')
        # if epoch % swa_start == 0 and epoch >= swa_start:
        #     optimizer = torch.optim.Adam(net.parameters(), lr=LR,  weight_decay=weight_decay)
        #     swa_scheduler = SWALR(optimizer, swa_lr=0.0001, anneal_strategy='linear')
        
        df_train = pd.DataFrame(columns=['sample_id', 'prediction', 'label'])
        net.train()

        patient_list = []
        prediction_list = []
        labels_list = []
        ofp.write("================================================================================================================\n")
        ofp.write(f"Epoch: {epoch:3>}/{epoch_num}\n")
        ofp.write("================================================================================================================\n")
        with tqdm(enumerate(trainloader, 0),total=len(trainloader),disable= not verbose) as t:
            o = 0
            for l in l_list_alw + l_list_deac:
                w = net.layers[l].weight
                w = w.view(w.shape[0],-1)
                s = w@w.T
                # print(s)
                n = w.square().sum(dim=1).sqrt()
                s = s/(n.view(-1,1)*n.view(1,-1))
                s = s*(1-torch.diag(torch.ones_like(n)))
                s1 = s.abs().sum()
                s1 = s1/(n.shape[0]*(n.shape[0]-1))
                print(s1.cpu().detach().numpy(),1/n.shape[0]*(n-.6).square().sum().item(),torch.round(n,decimals=3).cpu().detach().numpy())
                ofp.write(f"{s1.cpu().detach().numpy()},{1/n.shape[0]*(n-.6).square().sum().item()},{torch.round(n,decimals=3).cpu().detach().numpy()}\n")
            for j, (data,name,patients) in t:
                # inputs, labels, true_Labels = data['IEGM_seg'], data['label'], data['true_Labels']
                data = data.to(device,non_blocking=True)
                
                inputs, labels, true_Labels = trainset.gatherData(data), trainset.gatherLabels(data),trainset.gatherTrueLabels(data)
                total += labels.shape[0]
                optimizer.zero_grad()

                inputs = augmentations(inputs)
                outputs = net(inputs)
                if loss_function == 'Cross_Entropy':
                    loss = criterion(outputs, labels)

                    for l in l_list_alw:
                        w = net.layers[l].weight
                        w = w.view(w.shape[0],-1)
                        s = w@w.T
                        n = w.square().sum(dim=1).sqrt()
                        s = s/(n.view(-1,1)*n.view(1,-1))
                        s = s*(1-torch.diag(torch.ones_like(n)))
                        s = s.abs().sum()
                        s = s/(n.shape[0]*(n.shape[0]-1))
                    
                        loss += s
                        loss += 1/n.shape[0]*(n-.6).square().sum()

                    if epoch <100:
                        for l in l_list_deac:
                            w = net.layers[l].weight
                            w = w.view(w.shape[0],-1)
                            s = w@w.T
                            n = w.square().sum(dim=1).sqrt()
                            s = s/(n.view(-1,1)*n.view(1,-1))
                            s = s*(1-torch.diag(torch.ones_like(n)))
                            s = s.abs().sum()
                            s = s/(n.shape[0]*(n.shape[0]-1))
                        
                            loss += s
                            loss += 1/n.shape[0]*(n-.6).square().sum()
                            

                else:
                    onehot_labels = torch.nn.functional.one_hot(labels)
                    softmax_outputs = torch.nn.functional.softmax(outputs, dim = 1)
                    loss = soft_f1_loss(onehot_labels, softmax_outputs, fscore_beta)
                loss.backward()
                optimizer.step()

                
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                accuracy += correct / labels.shape[0]
                
                f1_labels = labels.detach().cpu().numpy()
                f1_prediction = predicted.detach().cpu().numpy()

                f1_labels[f1_labels==2]=0
                f1_prediction[f1_prediction==2]=0

                precision_i, recall_i, fscore_i, _ = precision_recall_fscore_support(f1_labels,f1_prediction , beta = 2, average='binary',zero_division=0)
                precision += precision_i
                recall += recall_i
                fscore += fscore_i
                i += 1
                for q in range(len(true_labels_idx)):
                    if trainset.true_labels_sum[q]==0:
                        continue
                    true_Lables[q] += torch.logical_and((predicted == labels),true_Labels==q).sum().item()/trainset.true_labels_sum[q]
                

                patient_list.append(patients)
                prediction_list.append(f1_prediction)
                labels_list.append(f1_labels)
      
                
                t.set_postfix({'Acc':accuracy / i,'loss': running_loss / i, 'F_B':fscore/i})
                t.set_description(f'Epoch: {epoch}/{epoch_num}, LR:{LR}')

            if epoch < swa_start:
                scheduler.step()
                lr_tmp = scheduler.get_last_lr()[0]
            else:
                swa_scheduler.step()
                lr_tmp = swa_scheduler.get_last_lr()[0]



            df_train = pd.DataFrame({'sample_id':np.concatenate(patient_list),'prediction':np.concatenate(prediction_list),'label':np.concatenate(labels_list)})

            patient_count = 0
            for patient in df_train["sample_id"].unique():
                df_patient = df_train[df_train["sample_id"] == patient]
                labels_patient = df_patient["label"].values
                predictions_patient = df_patient["prediction"].values
                patient_fscore = validation_Fscore(labels_patient, predictions_patient)
                if patient_fscore > 0.95:
                    patient_count += 1
            G_score = patient_count / len(df_train["sample_id"].unique())
            
            # lr_tmp = LR    
            # # lr_tmp = scheduler.get_last_lr()[0]
            # # scheduler.step()
            train_acc = accuracy / i
            train_precision = precision / i
            train_recall = recall / i
            train_fscore =  fscore / i
            Train_loss.append(running_loss / i)
            Train_acc.append(train_acc)
            Train_precison.append(train_precision)
            Train_recall.append(train_recall)
            Train_fscore.append(train_fscore)
            Train_G_score.append(G_score)
            Train_Full_score.append(.7*train_fscore+.3*G_score)
            epoch_list.append(epoch + 1)
            Learning_rate.append(lr_tmp)

            print('Train Acc: %.5f Train loss: %.5f, Train Fscore: %.5f, G_score: %.5f, Full Score: %.5f' % (train_acc, running_loss / i, fscore / i,G_score,.7*train_fscore+.3*G_score))
            ofp.write('Train Acc: %.5f Train loss: %.5f, Train Fscore: %.5f, G_score: %.5f, Full Score: %.5f\n' % (train_acc, running_loss / i, fscore / i,G_score,.7*train_fscore+.3*G_score))
            print(','.join(f'{a:>10}' for a in true_labels_idx.keys()))
            print(','.join(f'{true_Lables[a]*100:10.3}' for a in range(len(true_labels_idx))))
            running_loss = 0.0
            correct = 0.0
            total = 0.0
            i = 0.0
            running_loss_test = 0.0
            precision = 0.0
            recall = 0.0
            fscore = 0.0
            all_labels_test = []
            all_predicted_test = []
            true_Lables_test = np.zeros((len(true_labels_idx)))
            
            df_test = pd.DataFrame(columns=['sample_id', 'prediction', 'label'])


            patient_list = []
            prediction_list = []
            labels_list = []

            net.eval()
            for data_test,name,patients in tqdm(testloader,disable= not verbose):
                # IEGM_test, labels_test, true_Labels = data_test['IEGM_seg'], data_test['label'], data_test['true_Labels']
                data_test = data_test.to(device,non_blocking=True)
                IEGM_test, labels_test, true_Labels = testset.gatherData(data_test),testset.gatherLabels(data_test),testset.gatherTrueLabels(data_test)
                outputs_test = net(IEGM_test)
                _, predicted_test = torch.max(outputs_test.data, 1)
                total += labels_test.shape[0]
                correct += (predicted_test == labels_test).sum()
                all_labels_test.extend(list(labels_test.detach().cpu().numpy()))
                all_predicted_test.extend(list(predicted_test.detach().cpu().numpy()))
                
                if loss_function == 'Cross_Entropy':
                    loss_test= criterion(outputs_test, labels_test)
                else:
                    onehot_labels_test = torch.nn.functional.one_hot(labels_test)
                    softmax_outputs_test = torch.nn.functional.softmax(outputs_test, dim = 1)
                    loss_test = soft_f1_loss(onehot_labels_test, softmax_outputs_test, fscore_beta)
                running_loss_test += loss_test.item()

                for q in range(len(true_labels_idx)):
                    np.seterr(divide='ignore', invalid='ignore')
                    true_Lables_test[q] += torch.logical_and((predicted_test == labels_test),true_Labels==q).sum().item()/testset.true_labels_sum[q]


                patient_list.append(patients)


                i += 1
            test_acc = (correct / total).item()
            


            f1_labels = np.array(all_labels_test)
            f1_prediction = np.array(all_predicted_test)

            f1_labels[f1_labels==2]=0
            f1_prediction[f1_prediction==2]=0
            
            
            test_precision, test_recall, test_fscore, _ = precision_recall_fscore_support(f1_labels, f1_prediction, beta = 2, average='binary')
            
            prediction_list.append(f1_prediction)
            labels_list.append(f1_labels)

            df_test = pd.DataFrame({'sample_id':np.concatenate(patient_list),'prediction':np.concatenate(prediction_list),'label':np.concatenate(labels_list)})

            
            patient_count = 0
            for patient in df_test["sample_id"].unique():
                df_patient = df_test[df_test["sample_id"] == patient]
                labels_patient = df_patient["label"].values
                predictions_patient = df_patient["prediction"].values
                patient_fscore = validation_Fscore(labels_patient, predictions_patient)
                if patient_fscore > 0.95:
                    patient_count += 1
            G_score = patient_count / len(df_test["sample_id"].unique())



            print('Test Acc: %.5f Test Loss: %.5f Test Fscore: %.5f G_Score: %.5f, Full Score: %.5f'  % (test_acc, running_loss_test / i, test_fscore,G_score,.7*test_fscore+.3*G_score))
            ofp.write('Test Acc: %.5f Test Loss: %.5f Test Fscore: %.5f G_Score: %.5f, Full Score: %.5f\n'  % (test_acc, running_loss_test / i, test_fscore,G_score,.7*test_fscore+.3*G_score))
            print(','.join(f'{a:>10}' for a in true_labels_idx.keys()))
            print(','.join(f'{true_Lables_test[a]*100:10.3}' for a in range(len(true_labels_idx))))
        

            Test_G_score.append(G_score)
            Test_loss.append(running_loss_test / i)
            Test_acc.append(test_acc)
            Test_precison.append(test_precision)
            Test_recall.append(test_recall)
            Test_fscore.append(test_fscore)
            Test_Full_score.append(.7*test_fscore+.3*G_score)
            if  test_fscore > best_test_fscore:
                best_test_fscore = test_fscore
                best_train_fscore = train_fscore
                torch.save(net, result_save_path + save_name + f'.pkl')

            if (.7*test_fscore+.3*G_score) >= best_test_full_score:
                best_test_full_score = (.7*test_fscore+.3*G_score)
                torch.save(net, result_save_path + save_name + f'_hihest_full_score.pkl')
            torch.save(net, result_save_path + save_name + f'_epoch_{epoch}.pkl')

        
            df = pd.DataFrame()
            df['train_loss'] = Train_loss
            df['train_acc'] = Train_acc
            df['train_precision'] = Train_precison
            df['train_recall'] = Train_recall
            df['train_fscore'] = Train_fscore
            df['test_loss'] = Test_loss
            df['test_acc'] = Test_acc
            df['test_precision'] = Test_precison
            df['test_recall'] = Test_recall
            df['test_fscore'] = Test_fscore
            df['epoch'] = epoch_list
            df['learning_rate'] = Learning_rate
            df['weight_decay'] = weight_decay
            df['batch_size'] = BATCH_SIZE
            df['train_g_score'] = Train_G_score
            df['test_g_score'] = Test_G_score
            df['train_full_score'] = Train_Full_score
            df['test_full_score'] = Test_Full_score
            df.to_csv(result_save_path + save_name + f'.csv')
            ofp.flush()

    print('Finish training, Best-Test-Fscore: %0.5f, Best-Train-Fscore:%0.5f '  % (best_test_fscore, best_train_fscore ))
    
    return best_test_fscore, net

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=1000)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.00125)
    argparser.add_argument('--weight_decay', type=float, help='weight_decay', default=0.005) #best 0.005
    argparser.add_argument('--loss_function', type=str, help='F1 or Cross_Entropy', default='Cross_Entropy') # Cross_Entropy
    argparser.add_argument('--fscore-beta', type=int, help='fscore-beta', default=1)
    argparser.add_argument('--batchsz', type=int, help='total batchsz for traindb', default=32)
    argparser.add_argument('--cuda', type=int, default=0)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, 
                            default='./tinyml_contest_data_training/')
    argparser.add_argument('--path_indices', type=str, default='data_indices')
    argparser.add_argument('--result_save_path', type=str, default='./saved_models/experiments/')
    argparser.add_argument('--idx', type=int, default=-1)

    args = argparser.parse_args()

    device = torch.device("cuda:" + str(args.cuda))
    
    spezial = 'Gatech_simple_VFb'
    # spezial = 'Gatech_normal'

    print("device is --------------", device)
    best_f_score = 0
    best_net = None
    if args.idx == -1:
        for i in range(1,10):
            f_score, net = main(i)
            if f_score > best_f_score:
                best_f_score = f_score
                best_net = net
            print(f_score)
        date = datetime.now()
        date_string = date.strftime('%Y_%m_%d_%H%M')
        torch.save(best_net.state_dict(), args.result_save_path + f'date_string' + spezial + f'best_net_{i}.pkl')
    else:
        print(args.idx)
        main(args.idx,verbose=False)
        