import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import multiprocessing
import itertools
import argparse

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error as MAE
from scipy import stats
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm









###  
### Argument Parsing
parser = argparse.ArgumentParser(description='Parking Training')
parser.add_argument('--device', type=int, 
                    help='GPU Device Number')
parser.add_argument('--process', type=int, 
                    help='Number of Processes')


args = parser.parse_args()
devicenum = args.device
process = args.process

device = f"cuda:{str(devicenum)}" if torch.cuda.is_available() else "cpu"


###  
### Data Loading

finaltrain = pd.read_csv('finaltrain.csv')
finaltest = pd.read_csv('finaltest.csv')
sample_submission = pd.read_csv('sample_submission.csv')
meanstd = pd.read_csv('mean_std.csv')
meanstd = pd.DataFrame(np.array(meanstd), columns=['var', 'mean', 'std'])


###  
### Torch Dataset Define
class ParkingDataset(Dataset):
    
    def __init__(self, finaltrain):
        self.finaldata = finaltrain

    def __len__(self):
        return len(self.finaldata)

    def __getitem__(self, idx):
        target = [self.finaldata.iloc[idx]['등록차량수']]
        sample = np.array(self.finaldata[self.finaldata.columns.difference(['단지코드', '등록차량수'])].iloc[idx])

        return torch.Tensor(sample), torch.Tensor(target)

    
class SimpleDNN(nn.Module):

    def __init__(self, input_size, structure):
        super().__init__()
        
        self.layers = {}
        self.layers_input = nn.Linear(input_size, structure[0])
        self.num_layers = len(structure)
        
        for num in range(self.num_layers - 1):
            exec(f'self.layers{str(num)} = nn.Linear(structure[num], structure[num+1])')
        
        self.layers_output = nn.Linear(structure[-1], 1)

    def forward(self, x):
        # hidden, cell state init
        x = F.relu(self.layers_input(x))
        for num in range(self.num_layers - 1):
            exec(f'x = F.relu(self.layers{str(num)}(x))')
        
        x = self.layers_output(x)
            
        return x

    
def train(num_epochs, model, data_loader, val_loader, patience,
          criterion, optimizer, saved_dir, device):
    #     print('Start training..')
    best_loss = 9999999
    model.train()
    for epoch in range(num_epochs):
        if epoch == 0:
            early_stopping = EarlyStopping(patience=patience, path=saved_dir, verbose=False)
        else:
            early_stopping = EarlyStopping(patience=patience, best_score=best_score,
                                           counter=counter, path=saved_dir, verbose=False)

        for step, (sequence, target) in enumerate(data_loader):
            sequence = sequence.type(torch.float32)
            target = target.type(torch.float32)
            sequence, target = sequence.to(device), target.to(device)

            outputs = model(sequence)
            loss = criterion(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avrg_loss, mae = validation(model, val_loader, criterion, device)
        best_score, counter, finish = early_stopping(avrg_loss, model)

        if finish:
            model.load_state_dict(torch.load(saved_dir))
            model.eval()
            avrg_loss, b = validation(model, val_loader, criterion, device)
            break

    return best_score, mae


def validation(model, data_loader, criterion, device):
    b = []
    model.eval()
    with torch.no_grad():
        total_loss = 0
        cnt = 0
        for step, (sequence, target) in enumerate(data_loader):
            sequence = sequence.type(torch.float32)
            target = target.type(torch.float32)
            sequence, target = sequence.to(device), target.to(device)
            outputs = model(sequence)
            loss = criterion(outputs, target)
            b.append((outputs, target))
            total_loss += loss
            cnt += 1
        avrg_loss = total_loss / cnt
    #         print('Validation Average Loss: {:.4f}'.format(avrg_loss))
    model.train()
    return avrg_loss, b


class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""

    def __init__(self, patience=7, best_score=np.inf, counter=0, delta=0,
                 path=None, verbose=False):

        self.patience = patience
        self.verbose = verbose
        self.counter = counter
        self.best_score = best_score
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        if val_loss > self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
#                 print('Early Stopping Validated')
                self.early_stop = True

        else:
            self.save_checkpoint(val_loss, model)
            self.best_score = val_loss
            self.counter = 0

        return self.best_score, self.counter, self.early_stop

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if os.path.isfile(self.path):
            os.remove(self.path)
        if self.verbose:
            print(f'Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        
        
def MAE_func(a, b):
    a1 = torch.Tensor.cpu(a).numpy()
    b1 = torch.Tensor.cpu(b).numpy()
    
    return MAE(a1.squeeze(1), b1.squeeze(1))


def r3(value):
    return str(round(value, 3))


def trainsave(vars):
    batch_size, learning_rate, structure = vars
    
    structure_string = ''
    for node in structure:
        structure_string += str(node) + '_'
    structure_string = structure_string[:-1]
        
    for pth in os.listdir(f'vars/'):
        if f'ckpt_batch_{batch_size}_lr_{learning_rate}_structure_{structure_string}' in pth:
            return False

    # Base Parameters
    kf = KFold(n_splits=5, shuffle=True, random_state=777)
    patience = 50
    batch_size = batch_size
    num_epochs = 500
    learning_rate = learning_rate

    bs_box = []  # Best Scores
    mae_box = []
    for trainid, valid in kf.split(range(finaltrain.shape[0])):
        # Dataloader 구축; 5-fold validation
        parking_train = ParkingDataset(finaltrain.iloc[trainid])
        parking_val = ParkingDataset(finaltrain.iloc[valid])
        train_loader = DataLoader(parking_train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(parking_val, batch_size=batch_size, shuffle=False)


        # Training
        torch.manual_seed(7777)
        model = SimpleDNN(input_size=finaltrain.shape[1]-2, structure=structure)
        model = model.to(device)
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        saved_dir = f'saved/ckpt_batch_{batch_size}_lr_{learning_rate}_structure_{structure_string}.pt'

        best_score, mae = train(num_epochs, model, train_loader, val_loader, patience,
                                criterion, optimizer, saved_dir, device)

        bs_box.append(best_score)
        mae_box.append(np.nanmean([MAE_func(r[0], r[1]) for r in mae]))

    cvloss = np.mean([i.cpu().item() for i in bs_box])
    MAE = np.mean(mae_box)
    print(f'ckpt path: {saved_dir}\nBest CV_Loss: {r3(cvloss)}\nBest MAE: {r3(MAE)}')
    f = open(f'vars/ckpt_batch_{batch_size}_lr_{learning_rate}_structure_{structure_string} MSE {r3(cvloss)} RSQ {r3(MAE)}.txt', 'w')
    
    for i in mae_box:
        data = str(i) + "\n"
        f.write(data)
    f.close()



b = [2**i for i in range(2, 6)] # batch_size
l = [0.0003, 0.001, 0.003, 0.01] # learning_rate
s = [
    [400] * 6,
    [400] * 7,
    [400] * 8,
    [400] * 9,
] # structure

vars = list(itertools.product(b, l, s))

total_cnt = len(vars)
start = int(total_cnt / 3) * devicenum
if devicenum != 2:
    end = int(total_cnt / 3) * (devicenum + 1)
else:
    end = total_cnt


if __name__ == '__main__':
    
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(processes=process)
    pool.map(trainsave, vars[start:end])
    pool.close()
    pool.join()
