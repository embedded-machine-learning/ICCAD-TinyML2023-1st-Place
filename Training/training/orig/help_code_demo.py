import csv, torch, os
import numpy as np
from torch.utils.data import Dataset
import random
import tqdm


def ACC(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    total = sum(mylist)
    acc = (tp + tn) / total
    return acc


def PPV(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no VA segs for the patient, then ppv should be 1
    if tp + fn == 0:
        ppv = 1
    # for the case: there is some VA segs, but the predictions are wrong
    elif tp + fp == 0 and tp + fn != 0:
        ppv = 0
    else:
        ppv = tp / (tp + fp)
    return ppv


def NPV(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no non-VA segs for the patient, then npv should be 1
    if tn + fp == 0:
        npv = 1
    # for the case: there is some VA segs, but the predictions are wrong
    elif tn + fn == 0 and tn + fp != 0:
        npv = 0
    else:
        npv = tn / (tn + fn)
    return npv


def Sensitivity(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no VA segs for the patient, then sen should be 1
    if tp + fn == 0:
        sensitivity = 1
    else:
        sensitivity = tp / (tp + fn)
    return sensitivity


def Specificity(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no non-VA segs for the patient, then spe should be 1
    if tn + fp == 0:
        specificity = 1
    else:
        specificity = tn / (tn + fp)
    return specificity


def BAC(mylist):
    sensitivity = Sensitivity(mylist)
    specificity = Specificity(mylist)
    b_acc = (sensitivity + specificity) / 2
    return b_acc


def F1(mylist):
    precision = PPV(mylist)
    recall = Sensitivity(mylist)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def FB(mylist, beta=2):
    precision = PPV(mylist)
    recall = Sensitivity(mylist)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = (1+beta**2) * (precision * recall) / ((beta**2)*precision + recall)
    return f1

def stats_report(mylist):
    f1 = round(F1(mylist), 5)
    fb = round(FB(mylist), 5)
    se = round(Sensitivity(mylist), 5)
    sp = round(Specificity(mylist), 5)
    bac = round(BAC(mylist), 5)
    acc = round(ACC(mylist), 5)
    ppv = round(PPV(mylist), 5)
    npv = round(NPV(mylist), 5)

    output = str(mylist) + '\n' + \
             "F-1 = " + str(f1) + '\n' + \
             "F-B = " + str(fb) + '\n' + \
             "SEN = " + str(se) + '\n' + \
             "SPE = " + str(sp) + '\n' + \
             "BAC = " + str(bac) + '\n' + \
             "ACC = " + str(acc) + '\n' + \
             "PPV = " + str(ppv) + '\n' + \
             "NPV = " + str(npv) + '\n'

    # print("F-1 = ", F1(mylist))
    # print("F-B = ", FB(mylist))
    # print("SEN = ", Sensitivity(mylist))
    # print("SPE = ", Specificity(mylist))
    # print("BAC = ", BAC(mylist))
    # print("ACC = ", ACC(mylist))
    # print("PPV = ", PPV(mylist))
    # print("NPV = ", NPV(mylist))

    return output

def loadCSV(csvf):
    """
    return a dict saving the information of csv
    :param splitFile: csv file name
    :return: {label:[file1, file2 ...]}
    """
    dictLabels = {}
    with open(csvf) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)  # skip (filename, label)
        for i, row in enumerate(csvreader):
            filename = row[0]
            label = row[1]

            # append filename to current label
            if label in dictLabels.keys():
                dictLabels[label].append(filename)
            else:
                dictLabels[label] = [filename]
    return dictLabels


def txt_to_numpy(filename, row):
    with open(filename) as f:
        lines = f.readlines() 
    
    datamat = np.arange(row, dtype=np.float32)
    row_count = 0
    for line in lines:
        line = line.strip().split(' ')
        datamat[row_count] = line[0]
        row_count += 1

    return datamat


class ToTensor(object):
    def __call__(self, sample):
        text = sample['IEGM_seg']
        return {
            'IEGM_seg': torch.from_numpy(text),
            'label': sample['label']
        }

# class Normalize(torch):
#     def __init__(self) -> None:
#         pass
#     def __call__(self, sample):
#         text = sample['IEGM_seg']
#         return {
#             'IEGM_seg': torch.from_numpy(text),
#             'label': sample['label']
#         }
class IEGM_DataSET(Dataset):
    def __init__(self, root_dir, indice_dir, mode, size, transform=None, dataset_size = 100, augment = False, flip_peak = True, flip_time = False, add_noise = False, cache=False):
        self.root_dir = root_dir
        self.indice_dir = indice_dir
        self.size = size
        self.names_list = []
        self.transform = transform
        self.dataset_size = dataset_size
        self.augment = augment
        self.flip_peak = flip_peak
        self.flip_time = flip_time
        self.add_noise = add_noise
        self.IEGM_seg_cache = None
        self.cache = cache

        csvdata_all = loadCSV(os.path.join(self.indice_dir, mode + '_indice.csv'))

        for i, (k, v) in enumerate(csvdata_all.items()):
            self.names_list.append(str(k) + ' ' + str(v[0]))

        # Reduce the dataset to dataset_size percentage
        if self.dataset_size < 100:
            print()
            self.names_list = random.sample(self.names_list, int(len(self.names_list)*dataset_size/100))

        if cache:
            self.IEGM_seg_cache = np.zeros((len(self.names_list), 1, self.size, 1))

            for i in tqdm.tqdm(range(len(self.names_list)), desc='Caching data'):
                text_path = self.root_dir + self.names_list[i].split(' ')[0]
                IEGM_seg = txt_to_numpy(text_path, self.size).reshape(1, self.size, 1)
                self.IEGM_seg_cache[i] = IEGM_seg

    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, idx):
        if self.cache:
            IEGM_seg = self.IEGM_seg_cache[idx]
        else:
            text_path = self.root_dir + self.names_list[idx].split(' ')[0]

            if not os.path.isfile(text_path):
                print(text_path + 'does not exist')
                return None
            IEGM_seg = txt_to_numpy(text_path, self.size).reshape(1, self.size, 1)

        if self.augment:

            IEGM_seg_aug = np.copy(IEGM_seg)

            for i in range(len(IEGM_seg)):
                flip_p = random.random()
                flip_t = random.random()

                if flip_p < 0.5 and self.flip_peak:
                    IEGM_seg_aug[i] = -IEGM_seg[i]
                if flip_t < 0.5 and self.flip_time:
                    IEGM_seg_aug[i] = np.flip(IEGM_seg[i])
                if self.add_noise:
                    max_peak = IEGM_seg_aug[i].max() * 0.05
                    factor = random.random()
                    # factor = 1
                    noise = np.random.normal(0, factor * max_peak, (len(IEGM_seg_aug[i]), 1))
                    IEGM_seg_aug[i] = IEGM_seg_aug[i] + noise

            IEGM_seg = IEGM_seg_aug

        label = int(self.names_list[idx].split(' ')[1])
        #print(IEGM_seg.shape)
        sample = {'IEGM_seg': IEGM_seg, 'label': label, 'name': self.names_list[idx]}

        return sample


# class IEGM_DataSET():
#     def __init__(self, root_dir, indice_dir, mode, size, subject_id=None, transform=None):
#         self.root_dir = root_dir
#         self.indice_dir = indice_dir
#         self.size = size
#         self.names_list = []
#         self.transform = transform
#         self.data = {}

#         csvdata_all = loadCSV(os.path.join(self.indice_dir, mode + '_indice.csv'))

#         for i, (k, v) in enumerate(csvdata_all.items()):
#             # Check if the subject ID matches
#             if subject_id is not None and k.startswith(subject_id):
#                 self.names_list.extend([f"{k} {filename}" for filename in v])
#             elif subject_id is None:
#                 self.names_list.append(str(k) + ' ' + str(v[0]))

#     def __len__(self):
#         return len(self.names_list)

#     def __getitem__(self, idx):
#         text_path = self.root_dir + self.names_list[idx].split(' ')[0]

#         if not os.path.isfile(text_path):
#             print(text_path + ' does not exist')
#             return None 

#         if text_path not in self.data:
#             self.data[text_path]=txt_to_numpy(text_path, self.size).reshape(1, self.size,1)

#         IEGM_seg = self.data[text_path]
#         label = int(self.names_list[idx].split(' ')[1])
#         sample =  IEGM_seg
#         # print(sample.shape)
#         if self.transform:
#             sample = self.transform(sample)
#             sample = sample.reshape(1, self.size,1)
#         # print(sample.shape)

#         return sample, label


        IEGM_seg = txt_to_numpy(text_path, self.size).reshape(1, self.size, 1)
        label = int(self.names_list[idx].split(' ')[1])
        sample = {'IEGM_seg': IEGM_seg, 'label': label}

        return sample


def pytorch2onnx(net_path, net_name, size):
    net = torch.load(net_path, map_location=torch.device('cpu'))

    dummy_input = torch.randn(1, 1, size, 1)

    optName = str(net_name)+'.onnx'
    torch.onnx.export(net, dummy_input, optName, verbose=True)
    
# Special Loss Fucntion 
def soft_f1_loss(y_true, y_pred, beta = 1):
    tp = torch.sum(y_true*y_pred, 0)
    tn = torch.sum((1-y_true)*(1-y_pred), 0)
    fp = torch.sum((1-y_true)*y_pred, 0)
    fn = torch.sum(y_true*(1-y_pred), 0)

    ## True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    #TP_all = np.logical_and(prediction == 1, labels == 1)
    ## True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    #TN_all = np.logical_and(prediction == 0, labels == 0)
    ## False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    #FP_all = np.logical_and(prediction == 1, labels == 0)
    ## False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    #FN_all= np.logical_and(prediction == 0, labels == 1)

    p = tp/(tp+fp +1e-7)
    r = tp/(tp+fn+ 1e-7)

    f1 = (1+beta)*p*r / (beta*beta*p+r+1e-7)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
    return 1-torch.mean(f1)
