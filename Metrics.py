import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class IOU():
    def __init__(self, num_classes=21, ignore_class=255):
        self.num_classes = num_classes
        self.ignore_class = ignore_class

        self.intersection = np.zeros(num_classes, dtype=np.int32)
        self.union = np.zeros(num_classes, dtype=np.int32)
        self.ious = np.zeros(num_classes)

    def add(self, pred, target):
        intersection, union = [], []
        pred = pred.view(-1)
        target = target.data.view(-1)
        for class_ in range(self.num_classes):
            if class_ != self.ignore_class:
                pred_inds = pred == class_
                target_inds = target == class_
                inter = (pred_inds[target_inds]).long().sum()  # Cast to long to prevent overflows
                uni = pred_inds.long().sum() + target_inds.long().sum() - inter
                intersection.append(inter)
                union.append(uni)
            else:
                union.append(float('nan'))
                intersection.append(float('nan'))
        self.intersection += np.array(intersection,dtype=np.int32)
        self.union += np.array(union,dtype=np.int32)

    def get_mean_iou(self):
        for un, inter, (idx, iou_) in zip(self.union, self.intersection, enumerate(self.ious)):
            if un == 0.:
                self.ious[idx] = float('nan')
            else:
                self.ious[idx] = float(inter)/float(max(un, 1))
        notnan = self.ious[np.logical_not(np.isnan(self.ious))]
        mean_ious = notnan.sum()/len(notnan)
        return mean_ious

class seg_Accuracy():
    def __init__(self, ignore_class=255):
        self.ignore_class = ignore_class

        self.correct = 0L
        self.total = 0L
        self.accuracy = 0.0

    def add(self, pred, target):
        pred = pred.view(-1)
        target = target.data.view(-1)
        target_idx = target != self.ignore_class
        self.total += len(pred[target_idx])
        self.correct += long((pred[target_idx] == target[target_idx]).sum())

    def get_accuracy(self):
        self.accuracy = self.correct/float(self.total)
        return self.accuracy

class Conf_Mat():
    def __init__(self, classes, num_classes=21, ignore_class=255):
        self.num_classes = num_classes
        self.ignore_class = ignore_class
        self.conf_path = '/home/maxlotz/Thesis/Figs_pytorch/'
        self.classes = classes

        self.confusion = np.zeros((num_classes,num_classes), dtype=np.int32)

    def add(self, pred, target):
        pred = pred.view(-1).cpu().numpy()
        target = target.data.view(-1).cpu().numpy()
        pred_idx = pred != self.ignore_class
        target_idx = target != self.ignore_class
        not_ignore = np.logical_and(pred_idx, target_idx)
        pred = pred[not_ignore][None]
        target = target[not_ignore][None]
        data = np.concatenate((target,pred))
        uniq, uniq_counts = np.unique(data, return_counts=True, axis=1)
        self.confusion[uniq[0,:],uniq[1,:]] += uniq_counts

    def save(self, save_path):
        #create figure
        normalized = self.confusion/self.confusion.astype(np.float).sum(axis=1)[:, np.newaxis]
        plt.imshow(normalized, cmap=plt.cm.Blues)
        plt.title(save_path + ' Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=90, size=8)
        plt.yticks(tick_marks, self.classes, size=8)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(self.conf_path + save_path + '.png')

        #create csv
        TP = np.diag(self.confusion)
        actual_total = np.sum(self.confusion,0)
        pred_total = np.sum(self.confusion,1)
        recall = TP/actual_total.astype(np.float)
        precision = TP/pred_total.astype(np.float)
        #creates the matrix to be used in the CSV file
        csvdata = np.vstack([self.confusion,actual_total])
        csvdata = np.vstack([csvdata,recall])
        pred_total = np.concatenate((pred_total,np.zeros([2,])),0)
        precision = np.concatenate((precision,np.zeros([2,])),0)
        csvdata = np.hstack([csvdata,pred_total[None].T])
        csvdata = np.hstack([csvdata,precision[None].T])

        #saves as csv file
        df = pd.DataFrame(csvdata, index=self.classes + ['Actual Total','Recall'], columns=self.classes + ['Predicted Total','Precision'])
        df.to_csv(self.conf_path + save_path + '.csv')

def get_stats(dataset, n_classes):
    # set dataset param subtarct_mean to False first
    # ONLY USE ON SMALL DATASET <100,000 imgs, TAKES LONG TIME
    # returns mean and std of dataset as well as class frequency
    ln = len(dataset)
    sz = dataset[0][0].size()
    print "Loading dataset into tensor"
    print "-"*10
    tensor = torch.FloatTensor(ln, sz[0], sz[1], sz[2]).zero_()
    for idx, data in enumerate(dataset):
        if (idx % 100) == 0:
            print "loading image {} of {}".format(idx, ln)
        tensor[idx,:,:,:] = data[0]
        #for i in xrange(n_classes):

    print "-"*10
    print "Dataset loaded into tensor"
    B, G, R = tensor[:,0,:,:], tensor[:,1,:,:], tensor[:,2,:,:]
    mean = [B.mean(), G.mean(), R.mean()]
    std = [B.std(), G.std(), R.std()]
    return mean, std