import numpy as np
import torch
from torch import nn
import torch.utils.data as data_utils
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools

class MnistBags(data_utils.Dataset):
    def __init__(self,circ_files,lnc_files):
        self.circ_files=circ_files
        self.lnc_files=lnc_files
        self.seqs, self.train_bags_list, self.train_labels_list = self._create_bags()

    def seq_to_batch(self,seq,step=5,window_size=70):
        batch=len(seq)//step
        seqs=seq+seq
        return [self.base_to_id(seqs[ix*step:ix*step+window_size]) for ix in range(batch)]
    
    def base_to_id(self,bases):
        base_list=['A','T','G','C','N','H','B','D','V','R','M','S','W','Y','K']
        return [base_list.index(each) for each in bases]
    
    def pad_sequences(self,seqs,max_length=8000,unk_index=64):
        pad_seqs=[]
        for each in seqs:
            if len(each[0])>max_length:
                mid_index=max_length//2
                pad_seqs.append((each[0][:mid_index]+each[0][(len(each[0])-(max_length-mid_index)):],each[1]))
            else:
                pad_seqs.append(each)
        return pad_seqs
    
    def _create_bags(self):
        seqs=[]
        with open(self.circ_files) as f:
            for line in f:
                each=line.strip()
                seqs.append((each,[1]))
        with open(self.lnc_files) as f:
            for line in f:
                each=line.strip()
                seqs.append((each,[0]))
        #seqs=self.pad_sequences(seqs)
        bags_list = []
        labels_list = []
        seqs2=[]
        for each in seqs:
            seqs2.append(each[0])
            bags_list.append(np.array(self.seq_to_batch(each[0])))
            labels_list.append(each[1])
        
        return seqs2, bags_list, labels_list

    def __len__(self):
        return len(self.train_labels_list)

    def __getitem__(self, index):
        bag = self.train_bags_list[index]
        label = self.train_labels_list[index]
        return bag, label

circ_files_tr='circRNA_train.txt'
lnc_files_tr='lncRNA_train2.txt'
train_loader = MnistBags(circ_files_tr,lnc_files_tr)


class GatedAttention2(nn.Module):
    def __init__(self,initial_weight):
        super(GatedAttention2, self).__init__()
        self.L = 100
        self.D = 30
        self.K = 1     
        self.feature_extractor_part1 = nn.Sequential(
            nn.Embedding(15, 4),
            nn.LSTM(4, 150, 2,bidirectional=True),
        )
        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(150*2, self.L),
            nn.ReLU(),
        )
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )
        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )
        list(self.feature_extractor_part1.modules())[1].weight=Parameter(Variable(torch.Tensor(initial_weight).to(torch.float32), requires_grad=True))
    def forward(self, x):
        x=x.squeeze(0).permute(1,0)
        outputs, (_, _) = self.feature_extractor_part1(x)
        H = self.feature_extractor_part2(outputs[-1,:,:])  # NxL
        A_V = self.attention_V(H)  # NxD
        A = self.attention_weights(A_V) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, H)  # KxL
        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()
        return Y_prob, Y_hat, A
    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        return error, Y_hat
    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        return neg_log_likelihood, A

initial_weight=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
model=GatedAttention2(initial_weight)