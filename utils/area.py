import torch
from torch.autograd import Variable
from utils import util
from utils.util import *
import numpy as np


def testArea():
    print("--------------------------------I FUCK THE WORLD!!!!!!!!!!!!----------------------------------------------");


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def get_feature_mean(imbalanced_train_loader, model, cls_num_list):

    model.eval() 
    cls_num = len(cls_num_list) 
    device = next(model.parameters()).device 

    feature_dim = 256 

    feature_sum_0 = torch.zeros(cls_num, feature_dim, device=device)  
    feature_sum_1 = torch.zeros(cls_num, feature_dim, device=device)  


    with torch.no_grad():  
        for batch_idx, (inputs, target) in enumerate(imbalanced_train_loader):

            input_0 = inputs[0].to(device)
            input_1 = inputs[1].to(device)
            target = target.to(device)  

            target_one_hot = torch.zeros(target.size(0), cls_num, device=device).scatter_(1, target.view(-1, 1), 1)

            _, _, _, _, feature_0, center_0 = model(x=input_0, train=True)
            _, _, _, _, feature_1, center_1 = model(x=input_1, train=True)


            for idx in range(target.size(0)):
                cls_idx = target[idx] 
                feature_sum_0[cls_idx] += feature_0[idx]  
                feature_sum_1[cls_idx] += feature_1[idx] 

    img_num_tensor = torch.tensor(cls_num_list, device=device).unsqueeze(1).float()

    feature_mean_0 = feature_sum_0 / img_num_tensor  
    feature_mean_1 = feature_sum_1 / img_num_tensor  
    feature_mean_final = (feature_mean_0 + feature_mean_1) / 2  

    return feature_mean_final.detach()  

def calculate_eff_weight(imbalanced_train_loader, model, cls_num_list, train_propertype):
    model.eval()
    train_propertype = train_propertype.cuda()
    class_num = len(cls_num_list)
    eff_all = torch.zeros(class_num).float().cuda()


    with torch.no_grad():
        for i, (inputs, target) in enumerate(imbalanced_train_loader):
            target_one_hot = torch.zeros(target.size(0), class_num).scatter_(1, target.view(-1, 1), 1)
            target_one_hot = target_one_hot.cuda()
            inputs[0] = inputs[0].cuda()
            inputs[1] = inputs[1].cuda()
            input_var_0 = to_var(inputs[0], requires_grad=False)
            input_var_1 = to_var(inputs[1], requires_grad=False)
        
            output_0, output_cb_0, z0, p0, feature_0, center_0 = model(x=input_var_0,train=True)
            output_1, output_cb_1, z1, p1, feature_1, center_1 = model(x=input_var_1,train=True)
            features = (feature_0 + feature_1) / 2
            mu = train_propertype[target].detach()  # batch_size x d
            feature_bz = (features.detach() - mu)  # Centralization
            index = torch.unique(target)  # class subset
            index2 = target.cpu().numpy()
            eff = torch.zeros(class_num).float().cuda()

            for i in range(len(index)):  # number of class
                index3 = torch.from_numpy(np.argwhere(index2 == index[i].item()))
                index3 = torch.squeeze(index3)
                feature_juzhen = feature_bz[index3].detach()
                if feature_juzhen.dim() == 1:
                    eff[index[i]] = 1
                else:
                    _matrixA_matrixB = torch.matmul(feature_juzhen, feature_juzhen.transpose(0, 1))
                    _matrixA_norm = torch.unsqueeze(torch.sqrt(torch.mul(feature_juzhen, feature_juzhen).sum(axis=1)),
                                                    1)
                    _matrixA_matrixB_length = torch.mul(_matrixA_norm, _matrixA_norm.transpose(0, 1))
                    _matrixA_matrixB_length[_matrixA_matrixB_length == 0] = 1
                    r = torch.div(_matrixA_matrixB, _matrixA_matrixB_length)  # R
                    num = feature_juzhen.size(0)
                    a = (torch.ones(1, num).float().cuda()) / num  # a_T
                    b = (torch.ones(num, 1).float().cuda()) / num  # a
                    c = torch.matmul(torch.matmul(a, r), b).float().cuda()  # a_T R a
                    eff[index[i]] = 1 / c
            eff_all = eff_all + eff


        weights = eff_all
        weights = torch.where(weights > 0, 1 / weights, weights).detach()
        # weight
        fen_mu = torch.sum(weights)
        weights_new = weights / fen_mu
        weights_new = weights_new * class_num  # Eq.(14)
        weights_new = weights_new.detach()

        return weights_new