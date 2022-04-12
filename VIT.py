import os
import sys
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from vit_pytorch.vit_for_small_dataset import ViT
import sklearn.metrics as metrics
from utils import *



transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(224), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

batch_size = 512
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import timm
model = timm.create_model("vit_tiny_patch16_224_in21k", pretrained=True)
model.head = nn.Linear(model.head.in_features, 10)
model.to(DEVICE)

# model = ViT(image_size  = 32,
#             patch_size  = 16,
#             num_classes = 10,
#             dim         = 1024,
#             depth       = 6,
#             heads       = 16,
#             mlp_dim     = 2048,
#             dropout     = 0.1,
#             emb_dropout = 0.1).to(DEVICE)



print(f'WORKING WITH {DEVICE}', file = sys.stderr)

epochs          = 100
learning_rate   = 0.0001
save_model_name = 'ViT'
model_save_path = "./saved_models/" + save_model_name + "/"
if not os.path.isdir(model_save_path):
    os.mkdir(model_save_path)

criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

SOTA_ACC_VAL      , SOTA_LOSS_VAL      = 0 , 0
bestResult_pred_np, bestResult_anno_np = [], []
bestModel                              = model

for epoch in range(epochs):
    model.train()
    LOSS_TRACE_FOR_TRAIN, LOSS_TRACE_FOR_VAL = [], []
    for idx, batch in enumerate(trainloader):
        optimizer.zero_grad()
        X_train, Y_train = batch
        X_train, Y_train = X_train.to(DEVICE), Y_train.to(DEVICE)

        Y_pred_train = model(X_train)
        Y_train = Y_train.squeeze(-1)

        LOSS_train = criterion(Y_pred_train, Y_train)

        LOSS_TRACE_FOR_TRAIN.append(LOSS_train.cpu().detach().numpy())
        LOSS_train.backward()
        optimizer.step()
    
    
    with torch.no_grad():
        model.eval()
        Result_pred_val, Result_anno_val = [], []
        for idx, batch in enumerate(testloader):
            X_val, Y_val = batch
            X_val, Y_val = X_val.to(DEVICE), Y_val.to(DEVICE)

            Y_pred_val = model(X_val)
            Y_val      = Y_val.squeeze(-1)
            
            LOSS_val = criterion(Y_pred_val, Y_val)
            LOSS_TRACE_FOR_VAL.append(LOSS_val.cpu().detach().numpy())

            Y_pred_val_np  = Y_pred_val.to('cpu').detach().numpy()
            Y_pred_val_np  = np.argmax(Y_pred_val_np, axis=1).squeeze()
            Y_val_np       = Y_val.to('cpu').detach().numpy().reshape(-1, 1).squeeze()     
            
            Result_pred_val = np.hstack((Result_pred_val, Y_pred_val_np))
            Result_anno_val = np.hstack((Result_anno_val, Y_val_np))
    
    Result_pred_np = np.array(Result_pred_val)
    Result_anno_np = np.array(Result_anno_val)
    Result_pred_np = np.reshape(Result_pred_np, (-1, 1))
    Result_anno_np = np.reshape(Result_anno_np, (-1, 1))
    
    ACC_VAL        = metrics.accuracy_score(Result_anno_np, Result_pred_np)
    AVG_LOSS_TRAIN = np.average(LOSS_TRACE_FOR_TRAIN)
    AVG_LOSS_VAL   = np.average(LOSS_TRACE_FOR_VAL)
    
    if ACC_VAL > SOTA_ACC_VAL:
        SOTA_ACC_VAL       = ACC_VAL
        SOTA_LOSS_VAL      = AVG_LOSS_VAL
        bestModel          = model
        bestResult_pred_np = Result_pred_np
        bestResult_anno_np = Result_anno_np
        
    printLearningData(epoch, epochs, AVG_LOSS_TRAIN, AVG_LOSS_VAL, ACC_VAL)
    
get_metrics(pred    = bestResult_pred_np,
            anno    = bestResult_anno_np,
            n_label = 10)

torch.save(bestModel.state_dict(), f'{model_save_path}{save_model_name}_CIFAR10_{SOTA_ACC_VAL*100:.2f}.pth')