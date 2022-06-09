from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.feature_selection import SelectKBest,chi2,f_classif
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import pearsonr
import cv2
import skimage.feature
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from matplotlib import pyplot as plt
import numpy as np
from cellnet import cellulav2,cellula
import torch as t
from tools import imshow,clahe_equalized,dataset_normalized,adjust_gamma,SBS
import joblib
from sklearn.model_selection import cross_val_score,GridSearchCV
import time
#%%
net = cellulav2()
mnet = cellula()#多分辨率网络
cuda_gpu = t.cuda.is_available()
gpus = [0]
if (cuda_gpu):
    net = t.nn.DataParallel(net, device_ids=gpus).cuda()
    mnet = t.nn.DataParallel(mnet, device_ids=gpus).cuda()
#%%
checkpoint = t.load("model_temp.ph")
mnet.load_state_dict(checkpoint['net']) # 用这个网络结构去load这个存的模型，加载进去
mnet.eval()
#%%
def getHist(path,len,mode,p,r,imgH,imgL):
    image = cv2.imread(path)
    image = image[:, :, 0]
    image2 = dataset_normalized(image)
    image2 = clahe_equalized(image2)
    image2 = adjust_gamma(image2, 1.2)
    netInput = t.from_numpy(image2.reshape(1, 1, imgH, imgL)).to("cuda").float()
    if mode==1:
        image_lbp = skimage.feature.local_binary_pattern(
            image, p, r, method='uniform');image_lbp = image_lbp.astype(np.uint8)
        temp = image_lbp
    elif mode==2:
        image_lbp = skimage.feature.local_binary_pattern(
            image, p, r, method='uniform');image_lbp = image_lbp.astype(np.uint8)
        output = net(netInput)
        output = output > 0.5
        output = output.cpu().detach().numpy().astype(np.uint8)
        temp = image_lbp+output*(image_lbp.max()+1)
    elif mode==3:
        image_lbp = skimage.feature.local_binary_pattern(
            image, p, r, method='uniform');image_lbp = image_lbp.astype(np.uint8)
        output, a, b, c = mnet(netInput)
        a = a.cpu().detach().numpy()
        b = b.cpu().detach().numpy()
        c = c.cpu().detach().numpy()
        baseN = image_lbp.max()+1
        temp = image_lbp + baseN*4 * a + baseN*2 * b + baseN * c#56  28  14
    elif mode==9:
        image_lbp_8_1 = skimage.feature.local_binary_pattern(
            image, 8, 1, method='uniform');image_lbp_8_1 = image_lbp_8_1.astype(np.uint8)
        image_lbp_16_2 = skimage.feature.local_binary_pattern(
            image, 16, 2, method='uniform');image_lbp_16_2 = image_lbp_16_2.astype(np.uint8)
        image_lbp_24_3 = skimage.feature.local_binary_pattern(
            image, 24, 3, method='uniform');image_lbp_24_3 = image_lbp_24_3.astype(np.uint8)
        image_lbp_8_3 = skimage.feature.local_binary_pattern(
            image, 8,3, method='uniform');image_lbp_8_3 = image_lbp_8_3.astype(np.uint8)
        image_lbp_12_5 = skimage.feature.local_binary_pattern(
            image, 12,5, method='uniform');image_lbp_12_5 = image_lbp_12_5.astype(np.uint8)
        image_lbp_16_7 = skimage.feature.local_binary_pattern(
            image, 16, 7, method='uniform');image_lbp_16_7 = image_lbp_16_7.astype(np.uint8)
        output, a, b, c = mnet(netInput)
        output = output > 0.5
        output = output.cpu().detach().numpy()
        a = a.cpu().detach().numpy()
        b = b.cpu().detach().numpy()
        c = c.cpu().detach().numpy()
        temp1 = image_lbp_8_1+40*a+20*b+10*c;temp1 = temp1.reshape(imgH, imgL);hist1 = cv2.calcHist([temp1.astype(np.uint8)], [0], None, [8*10], [0,8*10]);hist1 = np.reshape(hist1,(1,8*10))
        temp2 = image_lbp_16_2+4*18*a+2*18*b+18*c;temp2 = temp2.reshape(imgH, imgL);hist2 = cv2.calcHist([temp2.astype(np.uint8)], [0], None, [8*18], [0,8*18]);hist2 = np.reshape(hist2,(1,8*18))
        temp3 = image_lbp_24_3+4*26*a+2*26*b+26*c;temp3 = temp3.reshape(imgH, imgL);hist3 = cv2.calcHist([temp3.astype(np.uint8)], [0], None, [8*26], [0,8*26]);hist3 = np.reshape(hist3,(1,8*26))
        temp4 = image_lbp_8_3+4*10*a+20*b+10*c;temp4 = temp4.reshape(imgH, imgL);hist4 = cv2.calcHist([temp4.astype(np.uint8)], [0], None, [8*10], [0,8*10]);hist4 = np.reshape(hist4,(1,8*10))
        temp5 = image_lbp_12_5+4*14*a+2*14*b+14*c;temp5 = temp5.reshape(imgH, imgL);hist5 = cv2.calcHist([temp5.astype(np.uint8)], [0], None, [8*14], [0,8*14]);hist5 = np.reshape(hist5,(1,8*14))
        temp6 = image_lbp_16_7+4*18*a+2*18*b+18*c;temp6 = temp6.reshape(imgH, imgL);hist6 = cv2.calcHist([temp6.astype(np.uint8)], [0], None, [8*18], [0,8*18]);hist6 = np.reshape(hist6,(1,8*18))
        hist = np.concatenate((hist1,hist2,hist3,hist4,hist5,hist6),axis=1)
        return hist
    elif mode==10:
        image_lbp_8_1 = skimage.feature.local_binary_pattern(
            image, 8, 1, method='uniform');image_lbp_8_1 = image_lbp_8_1.astype(np.uint8)
        image_lbp_16_2 = skimage.feature.local_binary_pattern(
            image, 16, 2, method='uniform');image_lbp_16_2 = image_lbp_16_2.astype(np.uint8)
        output, a, b, c = mnet(netInput)
        output = output > 0.5
        output = output.cpu().detach().numpy()
        a = a.cpu().detach().numpy()
        b = b.cpu().detach().numpy()
        c = c.cpu().detach().numpy()
        temp1 = image_lbp_8_1+40*a+20*b+10*c;temp1 = temp1.reshape(imgH, imgL);hist1 = cv2.calcHist([temp1.astype(np.uint8)], [0], None, [8*10], [0,8*10]);hist1 = np.reshape(hist1,(1,8*10))
        temp2 = image_lbp_16_2+4*18*a+2*18*b+18*c;temp2 = temp2.reshape(imgH, imgL);hist2 = cv2.calcHist([temp2.astype(np.uint8)], [0], None, [8*18], [0,8*18]);hist2 = np.reshape(hist2,(1,8*18))
        hist = np.concatenate((hist1,hist2),axis=1)
        return hist
    elif mode==11:
        image_lbp_8_1 = skimage.feature.local_binary_pattern(
            image, 8, 1, method='uniform');image_lbp_8_1 = image_lbp_8_1.astype(np.uint8)
        image_lbp_16_2 = skimage.feature.local_binary_pattern(
            image, 16, 2, method='uniform');image_lbp_16_2 = image_lbp_16_2.astype(np.uint8)
        image_lbp_24_3 = skimage.feature.local_binary_pattern(
            image, 24, 3, method='uniform');image_lbp_24_3 = image_lbp_24_3.astype(np.uint8)
        image_lbp_8_3 = skimage.feature.local_binary_pattern(
            image, 8,3, method='uniform');image_lbp_8_3 = image_lbp_8_3.astype(np.uint8)
        output, a, b, c = mnet(netInput)
        output = output > 0.5
        output = output.cpu().detach().numpy()
        a = a.cpu().detach().numpy()
        b = b.cpu().detach().numpy()
        c = c.cpu().detach().numpy()
        temp1 = image_lbp_8_1+40*a+20*b+10*c;temp1 = temp1.reshape(imgH, imgL);hist1 = cv2.calcHist([temp1.astype(np.uint8)], [0], None, [8*10], [0,8*10]);hist1 = np.reshape(hist1,(1,8*10))
        temp2 = image_lbp_16_2+4*18*a+2*18*b+18*c;temp2 = temp2.reshape(imgH, imgL);hist2 = cv2.calcHist([temp2.astype(np.uint8)], [0], None, [8*18], [0,8*18]);hist2 = np.reshape(hist2,(1,8*18))
        temp3 = image_lbp_24_3+4*26*a+2*26*b+26*c;temp3 = temp3.reshape(imgH, imgL);hist3 = cv2.calcHist([temp3.astype(np.uint8)], [0], None, [8*26], [0,8*26]);hist3 = np.reshape(hist3,(1,8*26))
        temp4 = image_lbp_8_3+4*10*a+20*b+10*c;temp4 = temp4.reshape(imgH, imgL);hist4 = cv2.calcHist([temp4.astype(np.uint8)], [0], None, [8*10], [0,8*10]);hist4 = np.reshape(hist4,(1,8*10))
        hist = np.concatenate((hist1,hist2,hist3,hist4),axis=1)
        return hist
    temp = temp.reshape(imgH, imgL)
    hist = cv2.calcHist([temp.astype(np.uint8)], [0], None, [len], [0, len])  # 分箱数，范围+1
    hist = np.reshape(hist,(1,len))
    return hist


def OutexDataLoad(Flen,Fmode,p,r):
    OutexTrainx = np.zeros((4560, Flen))
    OutexTestx = np.zeros((4560, Flen))
    OutexTesty = [i for i in range(24) for j in range(190)]
    OutexTrainy = [i for i in range(24) for j in range(20)]
    OutexTrainy.extend([i for i in range(24) for j in range(170)])
    i_1 = 0
    for i_class in range(24):#24个类
        for i_in_class in range(20):#一个类20张用来训练
            image_id = i_class*20+i_in_class
            path = './dataset/Outex_TC_00012/'+'%06d.ras'%(image_id)
            time1 = time.time()
            OutexTrainx[i_1,:] = getHist(path,Flen,Fmode,p,r,128,128)
            time2 = time.time()
            timedata.append(time2-time1)
            i_1 +=1
    i_2 = 0
    for i_class in range(24):#24个类
        for i_in_class in range(180):
            if i_in_class <=84:
                image_id = 480 + i_class * 180 + i_in_class
                path = './dataset/Outex_TC_00012/'+'%06d.ras'%(image_id)
                time1 = time.time()
                OutexTrainx[i_1,:] = getHist(path,Flen,Fmode,p,r,128,128)
                time2 = time.time()
                timedata.append(time2-time1)
                i_1 +=1;
                #print(i_1)
                image_id = 4800+i_class * 180 + i_in_class
                path = './dataset/Outex_TC_00012/'+'%06d.ras'%(image_id)
                OutexTrainx[i_1,:] = getHist(path,Flen,Fmode,p,r,128,128)
                i_1 +=1;
                #print(i_1)
            else:
                image_id = 480 + i_class * 180 + i_in_class
                path = './dataset/Outex_TC_00012/' + '%06d.ras' % (image_id)
                time1 = time.time()
                OutexTestx[i_2, :] = getHist(path, Flen, Fmode, p, r, 128, 128)
                time2 = time.time()
                timedata.append(time2 - time1)
                i_2 += 1;
                image_id = 4800 + i_class * 180 + i_in_class
                path = './dataset/Outex_TC_00012/' + '%06d.ras' % (image_id)
                OutexTestx[i_2, :] = getHist(path, Flen, Fmode, p, r, 128, 128)
                i_2 += 1;
    return OutexTrainx,OutexTrainy,OutexTestx,OutexTesty


def nnclassifier(trainx,trainy,testx,testy,Flen):
    nn = MLPClassifier(max_iter=10000, hidden_layer_sizes=400)
    nn.fit(trainx, trainy)
    time1 = time.time()
    test_score = nn.score(testx, testy)
    time2 = time.time()
    timedata.append(time2 - time1)
    return test_score

timedata=[]
acc = []
te = []
tm = []
Fopt = []
for Fmode,Flen,p,r in [(1,10,8,1),(1,14,12,3),(1,18,16,5),(2,20,8,1),(2,28,12,3),(2,36,16,5),(3,80,8,1),(3,14*8,12,3),(3,18*8,16,5),(10,8*(10+18),0,0),(11,8*(10+18+26+10),0,0),(9,8*96,0,0)]:
    trainx,trainy,testx,testy = OutexDataLoad(Flen,Fmode,p,r)
    te.append(np.mean(np.array(timedata)))
    timedata = []
    score =  nnclassifier(trainx, trainy, testx, testy, Flen)
    tm.append(np.mean(np.array(timedata)))
    timedata=[]
    print('OTC at param:' + str(Fmode) + ' ' + str(Flen) + ' ' + str(p) + ' ' + str(r) + ' score is :' + str(score))
    acc.append(score)
print(acc)