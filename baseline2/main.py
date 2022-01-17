import pandas as pd
import numpy as np
import os
import librosa
import librosa.display
import sklearn.preprocessing
import warnings
import torch
from torch.utils.data import Dataset
import math
import argparse
import nsml
from nsml import DATASET_PATH
from Custom import CustomDataset, CNN
from tqdm import tqdm

print('torch version: ', torch.__version__)

#경고 무시
warnings.filterwarnings(action='ignore') 

# GPU 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device: ', device)

# 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(42)
else: torch.manual_seed(42)

def full_path_df(root_path, df, mode='train') :
    full_path_df = pd.DataFrame({'left_path' : root_path + '/' + df['file_name'],
                     'right_path' : root_path + '/' + df['file_name_']})
    if mode == 'train' :
        full_path_df['label'] = df['label']
        
    return full_path_df

# def max_len_check(df) :
#     def len_check(path, sr=16000, n_mfcc=100, n_fft=400, hop_length=160) :
#         audio, sr = librosa.load(path, sr=sr)
#         mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
#         mfcc = sklearn.preprocessing.scale(mfcc, axis=1)

#         return mfcc.shape[1]

#     left_len = df['left_path'].apply(lambda x : len_check(x))
#     right_len = df['right_path'].apply(lambda x : len_check(x))

#     left_max_len = left_len.max()
#     right_max_len = right_len.max()

#     return (max(left_max_len, right_max_len))


def bind_model(model, parser):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *parser):
        os.makedirs(dir_name, exist_ok=True)
        save_dir = os.path.join(dir_name, 'checkpoint')
        torch.save(model.state_dict(), save_dir)

        # with open("max_len", "w") as f:
        #     f.write(str(max_len))

        print("저장 완료!")

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *parser):
        save_dir = os.path.join(dir_name, 'checkpoint')
        model.load_state_dict(torch.load(save_dir))

        # global max_len
        # with open("max_len", "r") as f:
        #     max_len = int(f.readline())

        print("로딩 완료!")

    def infer(test_path, **kwparser):
        test_data = pd.read_csv(os.path.join(test_path, 'test_data', 'test_data'))
        root_path = os.path.join(test_path, 'test_data', 'wav')
        test_df = full_path_df(root_path, test_data, mode='test')

        test_dataset = CustomDataset(test_df['left_path'], test_df['right_path'], label=None, mode='test')
        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

        model.eval()
        preds = []
        for batch in test_dataloader:
            X = batch['X'].to(device)
            with torch.no_grad():
                pred = model(X)
                pred = torch.tensor(torch.argmax(pred, axis=-1), dtype=torch.int32).cpu().numpy()
                preds += list(pred)
        
        prob = [1]*len(preds)
        
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(prob, pred)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 
        # 위의 포맷에서 prob은 리더보드 결과에 확률의 값은 영향을 미치지 않습니다(pred만 가져와서 채점). 
        # pred에는 예측한 binary 혹은 1에 대한 확률값을 넣어주시면 됩니다.
        
        return list(zip(prob, preds))

    nsml.bind(save=save, load=load, infer=infer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='nia_test')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--iteration', type=str, default='0')
    parser.add_argument('--pause', type=int, default=0)
    args = parser.parse_args()

    model = CNN().to(device)
    bind_model(model=model, parser=args)

    batch_size = 64
    

    if args.pause :
        nsml.paused(scope=locals())

    if args.mode == 'train' :
        train_path = os.path.join(DATASET_PATH, 'train')
        train_label = pd.read_csv(os.path.join(train_path, 'train_label'))
        root_path = os.path.join(train_path, 'train_data')
        train_df = full_path_df(root_path, train_label)

        # max_len = max_len_check(train_df)
        # print('max_len: ', max_len)
        train_dataset = CustomDataset(train_df['left_path'], 
                                    train_df['right_path'], 
                                    train_df['label'])

        learning_rate = 1e-5

        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        total_batch = math.ceil(len(train_dataset)/batch_size)

        print('학습 시작!')
        for epoch in range(args.epochs) :
            avg_cost = 0

            for batch in tqdm(train_dataloader):
                X = batch['X'].to(device)
                Y = batch['Y'].to(device)
                optimizer.zero_grad()
                hypothesis = model(X)
                cost = criterion(hypothesis, Y)
                cost.backward()
                optimizer.step()
                
                avg_cost += cost / total_batch
    
            print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, float(avg_cost)))

            nsml.save(epoch)
