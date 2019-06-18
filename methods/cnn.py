############ConvNets for Face Classification############
# Author: Haotian Tang                                                      # 
# E-Mail: kentang@sjtu.edu.cn                                            #
# Date: May, 2019                                                              #
#################################################

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicOp(nn.Module):
    def __init__(self, inc, outc):
        super(BasicOp, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class TransitionOp(nn.Module):
    def __init__(self, inc, outc):
        super(TransitionOp, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


# A small, plain CNN for face classification.
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        chn = 16
        self.stem = nn.Sequential(
            nn.Conv2d(3, chn, kernel_size=7, stride=2, padding=3), 
            nn.BatchNorm2d(chn),
            nn.ReLU(True),
            nn.AvgPool2d(2)
        ) # 24 x 24
        
        
        config = [
            ('B', chn), ('B', chn), # 24 x 24
            ('T', chn * 2), ('B', chn * 2), ('B', chn * 2), # 12 x 12
            ('T', chn * 4), ('B', chn * 4), ('B', chn * 4), # 6 x 6
            ('T', chn * 8), ('B', chn * 8), ('B', chn * 8) # 3 x 3
        ]
        
        main = []
        cur_chn = chn
        for layer_config in config:
            out_chn = layer_config[1]
            if layer_config[0] == 'B':
                main.append(BasicOp(cur_chn, out_chn))
            else:
                main.append(TransitionOp(cur_chn, out_chn))
            cur_chn = out_chn
        
        main.append(nn.AdaptiveAvgPool2d(1))
        self.net = nn.Sequential(*main)
        
        self.classifier = nn.Sequential(
            nn.Linear(chn * 8, chn * 8),
            nn.BatchNorm1d(chn * 8),
            nn.ReLU(True),
            nn.Linear(chn * 8, 2)
        )
        
    def forward(self, x):
        x = self.stem(x)
        x = self.net(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

# Provide a sckit-learn like interface.
class CNN:
    def __init__(self, epochs=20, batch_size=32, device='cuda:0', verbose=True, save_model=True):
        self.model = ConvNet()
        self.loss_fn = nn.CrossEntropyLoss()
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose
        self.save_model = save_model
    
    def fit(self, X_train, y_train):
        model = self.model.to(self.device, non_blocking=True)
        loss_fn = self.loss_fn.to(self.device, non_blocking=True)
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.epochs//2, gamma=0.5)
        model.train()
        
        
        batch_size = self.batch_size
        num_batches = (X_train.shape[0] + batch_size - 1) // batch_size
        
        # Train the CNN
        for epoch in range(1, self.epochs + 1):
            # Shuffle the data.
            shuffled_idx = np.random.permutation(X_train.shape[0])
            X_train = X_train[shuffled_idx, ...]
            y_train = y_train[shuffled_idx]
            
            # Train one epoch.
            scheduler.step()
            epoch_loss = 0.0
            epoch_correct = 0.0
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx+1) * batch_size, X_train.shape[0])
                batch_data = X_train[batch_start:batch_end, ...]
                batch_label = y_train[batch_start:batch_end]
                batch_data = torch.from_numpy(batch_data).float().to(self.device, non_blocking=True)
                batch_label = torch.from_numpy(batch_label).long().to(self.device, non_blocking=True)
                pred = model(batch_data)
                loss = loss_fn(pred, batch_label)
                epoch_loss += loss.item() * batch_data.shape[0]
                epoch_correct += torch.sum(pred.argmax(1) == batch_label).item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            epoch_loss = epoch_loss / X_train.shape[0]
            epoch_acc = epoch_correct / X_train.shape[0]
            if self.verbose:
                print("Epoch %d/%d training is done. Loss = %.4f, Acc = %.4f"%(epoch, self.epochs, epoch_loss, epoch_acc))
            
            if self.save_model:
                torch.save({'dic': model.state_dict(), 'epoch':epoch}, "CNN.pth")
    
    def predict(self, X_test, return_prob=False):
        model = self.model.to(self.device, non_blocking=True)
        if not os.path.exists("CNN.pth"):
            print("Please run fit function first!")
            return 0
        model.load_state_dict(torch.load("CNN.pth")['dic'])
        model.eval()
        batch_size = self.batch_size
        num_batches = (X_test.shape[0] + batch_size - 1) // batch_size
        outputs = []
        with torch.no_grad():
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx+1) * batch_size, X_test.shape[0])
                batch_data = X_test[batch_start:batch_end, ...]
                batch_data = torch.from_numpy(batch_data).float().to(self.device, non_blocking=True)
                if return_prob:
                    pred = F.softmax(model(batch_data), dim=1)[0][1]
                    return pred
                else:
                    pred = model(batch_data).argmax(1)
                pred = pred.cpu().numpy()
                outputs.append(pred)
        
        outputs = np.concatenate(outputs)
        return outputs
    
    
    def score(self, X_test, y_test):
        model = self.model.to(self.device, non_blocking=True)
        if not os.path.exists("CNN.pth"):
            print("Please run fit function first!")
            return 0
        model.load_state_dict(torch.load("CNN.pth")['dic'])
        model.eval()
        batch_size = self.batch_size
        num_batches = (X_test.shape[0] + batch_size - 1) // batch_size
        total_correct = 0.0
        with torch.no_grad():
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx+1) * batch_size, X_test.shape[0])
                batch_data = X_test[batch_start:batch_end, ...]
                batch_label = y_test[batch_start:batch_end]
                batch_data = torch.from_numpy(batch_data).float().to(self.device, non_blocking=True)
                batch_label = torch.from_numpy(batch_label).long().to(self.device, non_blocking=True)
                pred = model(batch_data)
                total_correct += torch.sum(pred.argmax(1) == batch_label).item()

        acc = total_correct / X_test.shape[0]
        print(acc)
        return acc