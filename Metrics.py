import torch
import numpy as np

def iou(pred, target, n_classes, ignore_class):
    # pred is of type: Long tensor, size BxWxH
    # target is of type: Variable, size BxWxH
    intersection, union = [], []
    pred = pred.view(-1)
    target = target.data.view(-1)
    
    for class_ in range(n_classes):
        if class_!=ignore_class:
            pred_inds = pred == class_
            target_inds = target == class_
            inter = (pred_inds[target_inds]).long().sum()  # Cast to long to prevent overflows
            uni = pred_inds.long().sum() + target_inds.long().sum() - inter
            intersection.append(inter)
            union.append(uni)
        else:
            union.append(float('nan'))
            intersection.append(float('nan'))
    return np.array(intersection), np.array(union)

'''
if union == 0:
    ious.append(float('nan'))
else:
    ious.append(float(intersection)/float(max(union, 1)))
'''