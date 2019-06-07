# in progress

from model.net import ResNet, BasicBlock
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

lambda_cls = 1
lambda_mask_mid = 1
lambda_edge_mid = 1
lambda_mask_final = 1

data = Variable(torch.randn([10, 150, 3, 3])).cuda()
mask = Variable(torch.ones([10, 1, 720, 1280]).type(torch.FloatTensor)).cuda()
edge = Variable(torch.ones([10, 1, 720, 1280]).type(torch.FloatTensor)).cuda()
label_x = Variable(torch.ones([10]).type(torch.LongTensor)).cuda()
label_y = Variable(torch.zeros([10]).type(torch.LongTensor)).cuda()


net = ResNet(BasicBlock, [2, 2, 2, 2]).cuda()

criterion_cls = nn.CrossEntropyLoss(reduction='mean').cuda()
criterion_image = nn.BCEWithLogitsLoss(reduction='mean').cuda()

x_cls, y_cls, mask_mid, edge_mid, mask_final = net(data, data)

loss_x_cls = criterion_cls(x_cls, label_x)
loss_y_cls = criterion_cls(y_cls, label_y)
loss_mask_mid = criterion_image(mask_mid, mask)
loss_edge_mid = criterion_image(edge_mid, edge)
loss_mask_final = criterion_image(mask_final, mask)

loss = lambda_cls*(loss_x_cls + loss_y_cls) + lambda_mask_mid*loss_mask_mid \
    + lambda_edge_mid*loss_edge_mid + lambda_mask_final*loss_mask_final


print(loss.item())
