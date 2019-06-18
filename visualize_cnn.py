import h5py
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from visualize import visualize_tsne

np.random.seed(1)


def PCAGPU(x_, K=100, device='cuda:0'):
    x_ = torch.from_numpy(x_).float().to(device)
    x = x_.transpose(0,1)@x_
    x = x_@torch.symeig(x, eigenvectors=True)[1][:K, ...].transpose(0,1)
    return x.cpu().numpy()

class GradCam:
    def __init__(self, model, device='cuda:0'):
        self.model = model.eval()
        self.feature = None
        self.gradient = None
        self.device = device

    def save_gradient(self, grad):
        self.gradient = grad

    def __call__(self, img_):
        h, w, _ = img_.shape
        img = img_.transpose(2,0,1)
        
        data = torch.from_numpy(img).float().to(self.device)
        data.requires_grad = True
        feature = data.unsqueeze(0)
        for name, module in self.model.named_children():
            if name == 'classifier':
                feature = feature.view(feature.size(0), -1)
            feature = module(feature)
            if name == 'stem':
                feature.register_hook(self.save_gradient)
                self.feature = feature
        
        classes = F.sigmoid(feature)
        one_hot, _ = classes.max(dim=-1)
        print(one_hot)
        self.model.zero_grad()
        one_hot.backward()
        
        weight = self.gradient.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        mask = F.relu((weight * self.feature).sum(dim=1)).squeeze(0)
        mask = cv2.resize(mask.data.cpu().numpy(), (w, h))
        mask = mask - np.min(mask)
        if np.max(mask) != 0:
            mask = mask / np.max(mask)
        heat_map = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET))
        cam = heat_map + np.float32((np.uint8(img.transpose((1, 2, 0)) * 1.0)))
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
        return cam

class CNNTsne:
    def __init__(self, model, device='cuda:0'):
        self.model = model.eval()
        self.shallow_feature = None
        self.final_feature = None
        self.device = device

    def save_gradient(self, grad):
        self.gradient = grad

    def __call__(self, img_):
        if len(img_.shape) == 3:
            h, w, _ = img_.shape
            img = img_.transpose(2,0,1)
            data = torch.from_numpy(img).float().to(self.device)
            feature = data.unsqueeze(0)
        else:
            _, h, w, _= img_.shape
            img = img_.transpose(0,3,1,2)
            feature = torch.from_numpy(img).float().to(self.device)
        
        for name, module in self.model.named_children():
            if name == 'classifier':
                feature = feature.view(feature.size(0), -1)
            feature = module(feature)
            if name == 'stem':
                feature.register_hook(self.save_gradient)
                self.shallow_feature = feature.reshape(feature.shape[0], -1).detach().cpu().numpy()
            if name == 'net':
                self.final_feature = feature.reshape(feature.shape[0], -1).detach().cpu().numpy()
        return self.shallow_feature, self.final_feature
        
if __name__ == '__main__':
    from cnn import *
    from data_processing.gen_bbox import base_dir
    model = ConvNet()
    model.load_state_dict(torch.load('CNN.pth')['dic'])
    model = model.cuda()
    
    gc = GradCam(model)
    
    hms = []
    rows = 3
    cols = 4
    dbfile = h5py.File('data_processing/fddb_positive_1.h5', 'r')
    data = dbfile['data'][...]
    dbfile.close()
    to_show = np.random.choice(data.shape[0], rows * cols)
    for row in range(rows):
        hms.append([])
        for col in range(cols):
            img = data[to_show[row*cols+col]]
            hm = (gc(img) * 255.).astype(np.uint8)
            hms[-1].append(hm)
    
    hms = np.array(hms).astype(np.uint8)
    hms = hms.transpose(0, 2, 1, 3, 4).reshape(rows*96, cols*96, 3)
    cv2.imwrite('gc.png', hms)
    
    
    """
    tsne = CNNTsne(model)
    dbfile = h5py.File('data_processing/fddb_positive_1.h5', 'r')
    pos_data = dbfile['data'][...][:500, ...]
    dbfile.close()
    
    dbfile = h5py.File('data_processing/fddb_negative_1.h5', 'r')
    neg_data = dbfile['data'][...][:500, ...]
    dbfile.close()
 
    data = np.vstack([pos_data, neg_data])
    shallow_f, deep_f = tsne(data)
    
    pca_shallow_data = PCAGPU(shallow_f)
    
    visualize_tsne(pca_shallow_data, 100, 'pca_shallow')
    visualize_tsne(deep_f, 128, 'pca_deep')
    """