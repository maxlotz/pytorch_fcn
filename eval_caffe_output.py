import cv2
import sys
import os
# suppresses non error or warning terminal messages
os.environ['GLOG_minloglevel'] = '2' 
import caffe
import numpy as np
import pandas as pd
import lmdb
from sklearn.utils import shuffle
from caffe.proto import caffe_pb2
from matplotlib import pyplot as plt

np.seterr(divide='ignore', invalid='ignore')

caffe.set_device(0)
caffe.set_mode_gpu()

IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

model_path = '/media/maxlotz/My Passport/Thesis/caffe_models/rgb_1_iter_5000.caffemodel'
prototxt_path = '/home/maxlotz/Thesis/prototxts/hsv_L_1_deploy.prototxt'
mean_path = '/home/maxlotz/Thesis/mean_files/rgb_1_mean.binaryproto'
image_path = '/home/maxlotz/Thesis/file_lists/rgb_test_1.txt'

csv_path = '/home/maxlotz/Thesis/Figs/ConfMatCSV/rgb_test_1.csv'
fig_path = '/home/maxlotz/Thesis/Figs/ConfMatFigs/rgb_1.png'
class_path = '/home/maxlotz/Thesis/file_lists/classes.txt'

with open(image_path, 'r') as file:
    data_list = file.read().splitlines()
    testfiles = [i.split(' ')[0]+' '+i.split(' ')[1] for i in data_list]
    testlabels = [i.split(' ')[2] for i in data_list]
    testlabels = [int(i) for i in testlabels]

with open(class_path,'r') as file:
    class_list = file.read().splitlines()

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    return cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

net = caffe.Net(prototxt_path, model_path, caffe.TEST)

mean_blob = caffe_pb2.BlobProto()
with open(mean_path) as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
	(mean_blob.channels, mean_blob.height, mean_blob.width))

# adds mean and reshapes data into W H C caffe format
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))

confusion = np.zeros([len(class_list),len(class_list)])
correct_count = 0

print "processing images..."
for i, (image, ground_truth) in enumerate(zip(testfiles, testlabels)):
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    out = out['prob']
    pred_class =  np.argmax(out)
    confusion[pred_class,ground_truth] +=1
    #print "predicted:" + "\t" + str(class_list[pred_class]) + "\t" + "actual:" + "\t" + str(class_list[correct_class])
    #cv2.imshow("image", img);
    #cv2.waitKey();
    if pred_class == ground_truth:
    	correct_count+=1.0

accuracy = correct_count/len(testfiles)
print "Accuracy:\t" + str(accuracy)
#True positives
TP = np.diag(confusion)
#Total number of images belonging to each class
actual_total = np.sum(confusion,0)
#Total number of times each class was predicted
pred_total = np.sum(confusion,1)

recall = TP/actual_total
precision = TP/pred_total

#creates the matrix to be used in the CSV file
csvdata = np.vstack([confusion,actual_total])
csvdata = np.vstack([csvdata,recall])
pred_total = np.concatenate((pred_total,np.zeros([2,])),0)
precision = np.concatenate((precision,np.zeros([2,])),0)
csvdata = np.hstack([csvdata,pred_total[None].T])
csvdata = np.hstack([csvdata,precision[None].T])

#saves as csv file
df = pd.DataFrame(csvdata, index=class_list + ['Actual Total','Recall'], columns=class_list + ['Predicted Total','Precision'])
df.to_csv(csv_path)

#Normalizes confusion matrix
confusion = confusion/np.sum(confusion,0)*255
confusion.astype(np.uint8)

plt.imshow(confusion,cmap='gray')
plt.savefig(fig_path)

#python eval_caffe_output.py hsv 0