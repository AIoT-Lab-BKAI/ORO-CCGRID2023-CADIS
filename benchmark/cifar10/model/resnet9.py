import torch.nn as nn
from utils.fmodule import FModule

def conv_block(in_channel, out_channel, pool=False):
    layers = [nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=1,stride=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)]

    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class Model(FModule):
  def __init__(self, in_channel=3, num_classes=10):
    super().__init__()

    self.conv1 = conv_block(in_channel,32)
    self.conv2 = conv_block(32,64,pool=True)
    self.res1 = nn.Sequential(conv_block(64,64),conv_block(64,64))

    self.conv3 = conv_block(64,128,pool=True)
    self.conv4 = conv_block(128,256,pool=True)
    self.res2 = nn.Sequential(conv_block(256,256),conv_block(256,256))

    self.pool = nn.MaxPool2d(4)
    self.dropout = nn.Dropout(0.2)
    self.classifier = nn.Linear(256, num_classes)
    
  def forward(self,x):
    out = self.conv1(x)
    out = self.conv2(out)
    out = self.res1(out) + out
    out = self.conv3(out)
    out = self.conv4(out)
    out = self.res2(out) + out
    out = self.pool(out)
    out = self.dropout(out.view(x.shape[0], -1))
    out = self.classifier(out)
    return out
  
  def pred_and_rep(self,x):
    out = self.conv1(x)
    out = self.conv2(out)
    out = self.res1(out) + out
    out = self.conv3(out)
    out = self.conv4(out)
    out = self.res2(out) + out
    out = self.pool(out)
    e = self.dropout(out.view(x.shape[0], -1))
    o = self.classifier(e)
    return o, e