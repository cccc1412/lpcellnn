# -*- coding: utf-8 -*
from turtle import down
import torch as t
from torch import nn
from tools import imshow
import numpy as np
class cellula(nn.Module):
    def __init__(self):
        super(cellula,self).__init__()
        #self.convWeight1 = nn.Parameter(t.randn(1,1,3,3))
        #self.convWeight2 = nn.Parameter(t.randn(1,1,3,3))
        #self.mybias = nn.Parameter(t.randn(1))
        #三个分辨率的AB核与偏置，共（9+1）*3 = 30个参数

        # self.convAx = nn.Parameter(t.cuda.FloatTensor([[[[0.2538,0.5022,0.07344],[0.17864,1.0622,0.24616],[0.2454,0.90548,0.52616]]]]))
        # self.convBx = nn.Parameter(t.cuda.FloatTensor([[[[ -0.0830,-3.15852,3.24172], [-2.0894,-9.34624,-2.0018], [6.8890,-1.53384,7.71856]]]]))
        # self.biasx = nn.Parameter(t.cuda.FloatTensor([-0.07002]))

        # self.convAu = nn.Parameter(
        #     t.cuda.FloatTensor([[[[0.2538,0.5022,0.07344],[0.17864,1.0622,0.24616],[0.2454,0.90548,0.52616]]]]))
        # self.convBu = nn.Parameter(t.cuda.FloatTensor(
        #     [[[[ -0.0830,-3.15852,3.24172], [-2.0894,-9.34624,-2.0018], [6.8890,-1.53384,7.71856]]]]))
        # self.biasu = nn.Parameter(t.cuda.FloatTensor([-0.07002]))
        #
        # self.convAd = nn.Parameter(
        #     t.cuda.FloatTensor([[[[0.2538,0.5022,0.07344],[0.1786,1.0622,0.24616],[0.2454,0.90548,0.52616]]]]))
        # self.convBd = nn.Parameter(t.cuda.FloatTensor(
        #     [[[[-0.0830, -3.15852, 3.24172], [-2.0894, -9.34624, -2.0018], [6.8890, -1.53384, 7.71856]]]]))
        # self.biasd = nn.Parameter(t.cuda.FloatTensor([-0.07002]))

        # self.convAx = nn.Parameter(t.randn(1,1,3,3))
        # self.convBx = nn.Parameter(t.randn(1,1,3,3))
        # self.biasx = nn.Parameter(t.randn(1))
        #
        # self.convAu = nn.Parameter(t.randn(1,1,3,3))
        # self.convBu = nn.Parameter(t.randn(1,1,3,3))
        # self.biasu = nn.Parameter(t.randn(1))
        #
        # self.convAd = nn.Parameter(t.randn(1,1,3,3))
        # self.convBd = nn.Parameter(t.randn(1,1,3,3))
        # self.biasd = nn.Parameter(t.randn(1))

        self.convAx = nn.Parameter(t.normal(mean=0,std=t.mul(t.ones(1,1,3,3),0.005)))
        self.convBx = nn.Parameter(t.normal(mean=0,std=t.mul(t.ones(1,1,3,3),0.005)))
        self.biasx = nn.Parameter(t.normal(mean=0,std=t.mul(t.ones(1),0.005)))

        self.convAu = nn.Parameter(t.normal(mean=0,std=t.mul(t.ones(1,1,3,3),0.005)))
        self.convBu = nn.Parameter(t.normal(mean=0,std=t.mul(t.ones(1,1,3,3),0.005)))
        self.biasu = nn.Parameter(t.normal(mean=0,std=t.mul(t.ones(1),0.005)))

        self.convAd = nn.Parameter(t.normal(mean=0,std=t.mul(t.ones(1,1,3,3),0.005)))
        self.convBd = nn.Parameter(t.normal(mean=0,std=t.mul(t.ones(1,1,3,3),0.005)))
        self.biasd = nn.Parameter(t.normal(mean=0,std=t.mul(t.ones(1),0.005)))



        # self.conv1x1_x.append(nn.Parameter(t.cuda.FloatTensor([[[[0]],[[0]],[[0]]]])))
        # self.conv1x1_x.append(nn.Parameter(t.cuda.FloatTensor([[[[0]],[[0]],[[0]]]])))
        # self.conv1x1_x.append(nn.Parameter(t.cuda.FloatTensor([[[[0]],[[0]],[[0]]]])))
        #
        # self.conv1x1_u.append(nn.Parameter(t.cuda.FloatTensor([[[[0]],[[0]],[[0]]]])))
        # self.conv1x1_u.append(nn.Parameter(t.cuda.FloatTensor([[[[0]],[[0]],[[0]]]])))
        # self.conv1x1_u.append(nn.Parameter(t.cuda.FloatTensor([[[[0]],[[0]],[[0]]]])))
        #
        # self.conv1x1_d.append(nn.Parameter(t.cuda.FloatTensor([[[[0]],[[0]],[[0]]]])))
        # self.conv1x1_d.append(nn.Parameter(t.cuda.FloatTensor([[[[0]],[[0]],[[0]]]])))
        # self.conv1x1_d.append(nn.Parameter(t.cuda.FloatTensor([[[[0]],[[0]],[[0]]]])))

        self.conv1x1_x_5 = nn.Parameter(t.normal(mean=0,std=t.mul(t.ones(1,3,1,1),0.005)))
        self.conv1x1_x_10 = nn.Parameter(t.normal(mean=0,std=t.mul(t.ones(1,3,1,1),0.005)))
        self.conv1x1_x_15 = nn.Parameter(t.normal(mean=0,std=t.mul(t.ones(1,3,1,1),0.005)))

        self.conv1x1_u_5 = nn.Parameter(t.normal(mean=0,std=t.mul(t.ones(1,3,1,1),0.005)))
        self.conv1x1_u_10 = nn.Parameter(t.normal(mean=0,std=t.mul(t.ones(1,3,1,1),0.005)))
        self.conv1x1_u_15 = nn.Parameter(t.normal(mean=0,std=t.mul(t.ones(1,3,1,1),0.005)))

        self.conv1x1_d_5 = nn.Parameter(t.normal(mean=0,std=t.mul(t.ones(1,3,1,1),0.005)))
        self.conv1x1_d_10 = nn.Parameter(t.normal(mean=0,std=t.mul(t.ones(1,3,1,1),0.005)))
        self.conv1x1_d_15 = nn.Parameter(t.normal(mean=0, std=t.mul(t.ones(1, 3, 1, 1), 0.005)))
        #self.conv1x1_d_15 = nn.Parameter(t.cuda.FloatTensor([[[[1/3]],[[1/3]],[[1/3]]]]))



        self.conv1x1 = nn.Parameter(t.normal(mean=0, std=t.mul(t.ones(1, 3, 1, 1), 0.005)))
        self.outBias = nn.Parameter(t.normal(mean=0,std=t.mul(t.ones(1),0.005)))

    def forward(self, x):
        #x = t.sub(x,t.min(x))#先映射到-1到1之间
        shape = list(x.size())
        x = t.mul(t.sub(x,t.min(x)),2/(t.max(x)-t.min(x)))
        x = t.sub(x,1)
        downSample = t.nn.functional.avg_pool2d(x,2)
        upSample = t.nn.functional.interpolate(x,size = [2*shape[2],2*shape[3]],mode='nearest')#有问题?

        BU = nn.functional.conv2d(x,self.convBx,bias = None,stride=1, padding=1, dilation=1, groups=1)
        BUdown = nn.functional.conv2d(downSample,self.convBd,bias = None,stride=1, padding=1, dilation=1, groups=1)
        BUup = nn.functional.conv2d(upSample,self.convBu,bias = None,stride=1, padding=1, dilation=1, groups=1)

        x = t.zeros(1,1,shape[2],shape[3])
        downSample = t.zeros(1,1,int(shape[2]/2),int(shape[3]/2))
        upSample = t.zeros(1,1,2*shape[2],2*shape[3])
        x = x.to('cuda')
        downSample = downSample.to('cuda')
        upSample = upSample.to('cuda')

        for i in range(20):
            if i ==5:
                newx = t.cat(tensors=[t.nn.functional.avg_pool2d(upSample,2),x,t.nn.functional.interpolate(downSample,[shape[2],shape[3]],mode = 'nearest')],dim=1)
                newUpsample = t.cat(tensors=[upSample,t.nn.functional.interpolate(x,[2*shape[2],2*shape[3]],mode = 'nearest'),t.nn.functional.interpolate(downSample,[2*shape[2],2*shape[3]],mode = 'nearest')],dim=1)
                newDownsample = t.cat(tensors=[t.nn.functional.avg_pool2d(upSample,4),t.nn.functional.avg_pool2d(x,2),downSample],dim=1)

                x = nn.functional.conv2d(newx,self.conv1x1_x_5,bias=None,stride=1, padding=0, dilation=1, groups=1)
                downSample = nn.functional.conv2d(newDownsample,self.conv1x1_d_5,bias=None,stride=1, padding=0, dilation=1, groups=1)
                upSample = nn.functional.conv2d(newUpsample,self.conv1x1_u_5,bias=None,stride=1, padding=0, dilation=1, groups=1)
            elif i==10:
                newx = t.cat(tensors=[t.nn.functional.avg_pool2d(upSample,2),x,t.nn.functional.interpolate(downSample,[shape[2],shape[3]],mode = 'nearest')],dim=1)
                newUpsample = t.cat(tensors=[upSample,t.nn.functional.interpolate(x,[2*shape[2],2*shape[3]],mode = 'nearest'),t.nn.functional.interpolate(downSample,[2*shape[2],2*shape[3]],mode = 'nearest')],dim=1)
                newDownsample = t.cat(tensors=[t.nn.functional.avg_pool2d(upSample,4),t.nn.functional.avg_pool2d(x,2),downSample],dim=1)


                x = nn.functional.conv2d(newx,self.conv1x1_x_10,bias=None,stride=1, padding=0, dilation=1, groups=1)
                downSample = nn.functional.conv2d(newDownsample,self.conv1x1_d_10,bias=None,stride=1, padding=0, dilation=1, groups=1)
                upSample = nn.functional.conv2d(newUpsample,self.conv1x1_u_10,bias=None,stride=1, padding=0, dilation=1, groups=1)

            elif i==15:
                newx = t.cat(tensors=[t.nn.functional.avg_pool2d(upSample,2),x,t.nn.functional.interpolate(downSample,[shape[2],shape[3]],mode = 'nearest')],dim=1)
                newUpsample = t.cat(tensors=[upSample,t.nn.functional.interpolate(x,[2*shape[2],2*shape[3]],mode = 'nearest'),t.nn.functional.interpolate(downSample,[2*shape[2],2*shape[3]],mode = 'nearest')],dim=1)
                newDownsample = t.cat(tensors=[t.nn.functional.avg_pool2d(upSample,4),t.nn.functional.avg_pool2d(x,2),downSample],dim=1)


                x = nn.functional.conv2d(newx,self.conv1x1_x_15,bias=None,stride=1, padding=0, dilation=1, groups=1)
                downSample = nn.functional.conv2d(newDownsample,self.conv1x1_d_15,bias=None,stride=1, padding=0, dilation=1, groups=1)
                upSample = nn.functional.conv2d(newUpsample,self.conv1x1_u_15,bias=None,stride=1, padding=0, dilation=1, groups=1)


            x = nn.functional.conv2d(x,self.convAx,bias = self.biasx,stride=1, padding=1, dilation=1, groups=1)
            x = t.add(x,BU)
            x = activeFun.apply(x)
            downSample = nn.functional.conv2d(downSample,self.convAd,bias=self.biasd,stride=1, padding=1, dilation=1, groups=1)
            downSample = t.add(downSample,BUdown)
            downSample = activeFun.apply(downSample)
            upSample = nn.functional.conv2d(upSample,self.convAu,bias=self.biasu,stride=1, padding=1, dilation=1, groups=1)
            upSample = t.add(upSample,BUup)
            upSample = activeFun.apply(upSample)


        downSample = t.nn.functional.interpolate(downSample,[shape[2],shape[3]],mode = 'nearest')
        upSample = t.nn.functional.avg_pool2d(upSample,2)
        output = t.cat(tensors=[upSample,x,downSample],dim=1)
        output = nn.functional.conv2d(output,self.conv1x1,bias=self.outBias,stride=1, padding=0, dilation=1, groups=1)
        output = activeFun.apply(output)
        #output = t.sigmoid(output)
        #output = t.mul(t.add(output, 1), 1 / 2)
        upSample = upSample>t.mean(upSample)
        x = x>t.mean(x)
        downSample = downSample > t.mean(downSample)

        upSample = upSample.int()
        x = x.int()
        downSample = downSample.int()

        return output,upSample,x,downSample

class cellulav2(nn.Module):#传统cellnn
    def __init__(self):
        super(cellulav2,self).__init__()

        # self.convAx = nn.Parameter(t.normal(mean=0,std=t.mul(t.ones(1,1,3,3),0.005)))
        # self.convBx = nn.Parameter(t.normal(mean=0,std=t.mul(t.ones(1,1,3,3),0.005)))
        # self.biasx = nn.Parameter(t.normal(mean=0,std=t.mul(t.ones(1),0.005)))

        self.convAx = nn.Parameter(t.cuda.FloatTensor([[[[0.2538,0.5022,0.07344],[0.17864,1.0622,0.24616],[0.2454,0.90548,0.52616]]]]))
        self.convBx = nn.Parameter(t.cuda.FloatTensor([[[[ -0.0830,-3.15852,3.24172], [-2.0894,-9.34624,-2.0018], [6.8890,-1.53384,7.71856]]]]))
        self.biasx = nn.Parameter(t.cuda.FloatTensor([-0.07002]))


    def forward(self, x):
        #x = t.sub(x,t.min(x))#先映射到-1到1之间
        x = t.mul(t.sub(x,t.min(x)),2/(t.max(x)-t.min(x))) #先映射到0到2
        x = t.sub(x,1)#再映射到-1到1
        BU = nn.functional.conv2d(x,self.convBx,bias = None,stride=1, padding=1, dilation=1, groups=1)
        shape =  list(x.size())
        x = t.zeros(shape[0],shape[1],shape[2],shape[3])
        x = x.to('cuda')

        for i in range(21):
            x = nn.functional.conv2d(x,self.convAx,bias = self.biasx,stride=1, padding=1, dilation=1, groups=1)
            x = t.add(x,BU)
            x = activeFun.apply(x)


        output = t.add(x,1)
        output = t.mul(output,1/2)
        return output

class activeFun(t.autograd.Function):
    @staticmethod
    def forward(ctx,input): #ctx  contex object
        ctx.save_for_backward(input)
        y = input.clone()
        y[input<-1] = -1
        y[input>1] = 1
        return y

    @staticmethod
    def backward(ctx,grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input<-1] = 0
        grad_input[input>1] = 0
        return grad_input


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, output, label,mask):

        mask = mask.ge(0.5)#确定没问题
        output = t.masked_select(output,mask)
        label = t.masked_select(label,mask) #01值
        #output = t.mul(t.add(output,1),1/2)
        N = label.size(0)
        smooth = 1

        output_flat = output.view(N, -1)#reshape
        tmaxet_flat = label.view(N, -1)

        intersection = t.mul(output_flat , tmaxet_flat)

        loss = 2 * (intersection.sum(0) + smooth) / (output_flat.sum(0) + tmaxet_flat.sum(0) + smooth)
        loss = 1 - loss.sum()

        return loss

class mseLoss(t.nn.Module):
    def __init__(self):
        super(mseLoss, self).__init__()
        self.criterion = nn.MSELoss()
    def forward(self,output,label,mask):#输入的是01值
        mask = mask.ge(0.5)#确定没问题
        # label = t.mul(label,2)#转为-1到1
        # label = t.sub(label,1)
        output = t.masked_select(output,mask)
        label = t.masked_select(label,mask)
        loss = self.criterion(output,label)
        return loss

class bceLoss(t.nn.Module):
    def __init__(self):
        super(bceLoss, self).__init__()
        self.criterion = nn.BCELoss()
    def forward(self,output,label,mask):#输入的是01值
        mask = mask.ge(0.5)#确定没问题
        output = t.masked_select(output,mask)
        label = t.masked_select(label,mask)
        loss = self.criterion(output,label)
        return loss

