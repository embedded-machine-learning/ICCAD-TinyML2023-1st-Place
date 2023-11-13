from orig.help_code_demo import *

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix


def validation_Fscore(labels,predictions):
    fp = 0.0
    tn = 0.0
    fn = 0.0
    tp = 0.0

    y_true_subject = np.array(labels)
    y_pred_subject = np.array(predictions)
    
    nva_idx = y_true_subject == 0
    va_idx = y_true_subject == 1
    
    fp += (len(y_true_subject[nva_idx]) - (y_pred_subject[nva_idx] == y_true_subject[nva_idx]).sum()).item()
    tn += ((y_pred_subject[nva_idx] == y_true_subject[nva_idx]).sum()).item()
    fn += (len(y_true_subject[va_idx]) - (y_pred_subject[va_idx] == y_true_subject[va_idx]).sum()).item()
    tp += ((y_pred_subject[va_idx] == y_true_subject[va_idx]).sum()).item()
    
    acc = (tn + tp) / (tn + fp + fn + tp)

    if (tp + fn == 0):
        precision = 1.0
    elif (tp + fp) != 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    if (tp + fn) != 0:
        sensitivity = tp / (tp + fn)
    else:
        sensitivity = 1.0

    if (fp + tn) != 0:
        FP_rate = fp / (fp + tn)
    else:
        FP_rate = 1.0

    # for the case: there is no VA segs for the patient
    if tp + fn == 0:
        PPV = 1
    # for the case: there is some VA segs
    elif tp + fp == 0 and tp + fn != 0:
        PPV = 0
    else:
        PPV = tp / (tp + fp)

    if (tn + fp) != 0:
        NPV = tn / (tn + fp)
    else:
        NPV = 1.0

    if (precision + sensitivity) != 0:
        F1_score = (2 * precision * sensitivity) / (precision + sensitivity)
    else:
        F1_score = 0.0

    if ((2 ** 2) * precision + sensitivity) != 0:
        F_beta_score = (1 + 2 ** 2) * (precision * sensitivity) / ((2 ** 2) * precision + sensitivity)
    else:
        F_beta_score = 0.0
    

    return F_beta_score

class FlipPeak:
    def __call__(self, x:torch.Tensor):
        """
        Args:
            x the time series, assumes axis 0 to be batch

        Returns:
            Tensor: Randomly Inverted Time Series.
        """
        sgn = torch.sign(torch.sign(torch.rand((x.shape[0],1,1,1),device=x.device)-.5)+.1)
        return x*sgn
    
class FlipSeg:
    def __call__(self, x:torch.Tensor):
        """
        Args:
            x the time series, assumes axis 0 to be batch

        Returns:
            Tensor: Randomly flipped time Series.
        """
        d =torch.rand((x.shape[0],1,1,1),device=x.device)
        return torch.where(d>=0.5,x,x.flip(2))

class AddNoise:
    def __call__(self, x:torch.Tensor):
        """
        Args:
            x the time series, assumes axis 0 to be batch

        Returns:
            Tensor: Randomly Noise added.
        """
        m = torch.amax(x,dim=2,keepdim=True) * 0.05
        noise = random.random()*m*torch.randn(x.shape,device=x.device)
        return x+noise
    

class Augmentations(nn.Module):
    def __init__(self, tansforms=[],*args,**kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.transforms = tansforms
    def forward(self,x):
        for f in self.transforms:
            x = f(x)
        return x
    
    

true_labels_idx={
'AFb' :     0,    
'AFt' :     1,    
'SR' :      2,
'SVT' :     3,  
'VFb' :     4,  
'VFt' :     5,  
'VPD' :     6,  
'VT' :      7,
}

class IEGM_DataSET(Dataset):
    def __init__(self, root_dir, indice_dir, mode, size, dataset_size = 100, cache=False, device='cpu'):
        self.root_dir = root_dir
        self.indice_dir = indice_dir
        self.size = size
        self.names_list = []
        self.dataset_size = dataset_size
        self.IEGM_seg_cache = None
        self.cache = cache
        self.device = device
        self.labels = []
        self.true_labels = []
        self.true_labels_sum = np.zeros(len(true_labels_idx))

        csvdata_all = loadCSV(os.path.join(self.indice_dir, mode + '_indice.csv'))

        for i, (k, v) in enumerate(csvdata_all.items()):
            self.names_list.append(str(k) + ' ' + str(v[0]))
            # if str(k).contains("-VFb-"):
            #     self.names_list.append(str(k) + ' ' + str(v[0]))
            #     self.names_list.append(str(k) + ' ' + str(v[0]))
            #     self.names_list.append(str(k) + ' ' + str(v[0]))
            #     self.names_list.append(str(k) + ' ' + str(v[0]))
            #     self.names_list.append(str(k) + ' ' + str(v[0]))
        


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
                self.true_labels.append(self.names_list[i].split('-')[-2])
                # print(self.true_labels[-1])

            self.IEGM_seg_cache = torch.tensor(self.IEGM_seg_cache,dtype=torch.float,device=self.device)
            self.true_labels = np.array([true_labels_idx[a] for a in self.true_labels])

            for q in range(len(true_labels_idx)):
                self.true_labels_sum[q] = np.sum(self.true_labels==q)

            # print(f'self.true_labels: {self.true_labels.shape}')
        self.labels = np.array([int(idx.split(' ')[1]) for idx in self.names_list])
        # self.labels[self.true_labels==3] = 2

        self.labels = torch.tensor(self.labels,dtype=torch.long,device=self.device)
        self.true_labels = torch.tensor(self.true_labels,device=self.device,dtype=torch.long)
        self.patients = [a.split(" ")[0].split("-")[0] for a in self.names_list] 


    def __len__(self):
        return len(self.names_list)

    def gatherData(self,idxs):
        return torch.index_select(self.IEGM_seg_cache,0,idxs)
    
    def gatherLabels(self,idxs):
        return torch.index_select(self.labels,0,idxs)
    
    def gatherTrueLabels(self,idxs):
        return torch.index_select(self.true_labels,0,idxs)

    def __getitem__(self, idx):
        return idx, self.names_list[idx],self.patients[idx]