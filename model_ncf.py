import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional
from torch.nn.init import xavier_normal_
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_users, num_items, layers=[64, 32, 16, 8]):
        super(Model, self).__init__()
        # Modules
        self.loss = torch.nn.BCELoss()
        self.MLP_Embedding_item = torch.nn.Embedding(num_items, layers[0]//2)
        self.MLP_Embedding_User = torch.nn.Embedding(num_users, layers[0]//2)
        self.Layer1 = torch.nn.Linear(layers[0], layers[1])
        self.Layer2 = torch.nn.Linear(layers[1], layers[2])
        self.Layer3 = torch.nn.Linear(layers[2], layers[3])
        self.Layer4 = torch.nn.Linear(layers[3], 1)
        self.init()

    def init(self):
        xavier_normal_(self.MLP_Embedding_User.weight.data)
        xavier_normal_(self.MLP_Embedding_item.weight.data)

    def forward(self, item_input, user_input):
        user_inputs = self.MLP_Embedding_User(user_input)
        item_inputs = self.MLP_Embedding_item(item_input)
        vector = torch.cat([item_inputs, user_inputs], dim=1).float()
        dout = torch.nn.functional.relu(self.Layer1(vector))
        dout = torch.nn.functional.relu(self.Layer2(dout))
        dout = torch.nn.functional.relu(self.Layer3(dout))
        prediction = torch.sigmoid(self.Layer4(dout))
        prediction = prediction.to(torch.float32)
        y = prediction.shape[0]
        prediction = prediction.reshape(y)
        return prediction

    def get_reg(self):
        return torch.sqrt(abs(self.Wo)**2).mean() + \
               torch.sqrt(abs(self.Wv)**2).mean() + torch.abs(self.Wq).mean()
               # torch.sqrt(abs(self.MLP_Embedding_User.weight.data)**2).mean() + torch.sqrt(abs(self.Q.weight.data)**2).mean() + \
               # torch.sqrt(abs(self.V.weight.data)**2).mean() + torch.sqrt(abs(self.Layer1.weight.data)**2).mean() + \
               # torch.sqrt(abs(self.Layer2.weight.data)**2).mean() + torch.sqrt(abs(self.Layer3.weight.data)**2).mean() + \
               # torch.sqrt(abs(self.Layer4.weight.data)**2).mean()


