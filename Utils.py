import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
plt.switch_backend('agg') #stops DISPLAY error when using SSH

def make_graph(log_name, graph_name):
	df = pd.read_csv(log_name + '.csv', header=0)
	data = df.values
	coldict = {column:idx for idx, column in enumerate(df.columns.tolist())}
	datadict = {'train': data[data[:,coldict['set']]==0], 'test': data[data[:,coldict['set']]==1]}
	batches_per_epoch = np.max(datadict['train'][:,coldict['batch']])
	get_batch  = lambda set_: datadict[set_][:,coldict['batch']] \
							+ datadict[set_][:,coldict['epoch']] \
							* batches_per_epoch
	batchno = map(get_batch, ['train', 'test'])
	
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.plot(batchno[0], datadict['train'][:,coldict['loss']], 'r', label='Train loss')
	ax1.plot(batchno[1], datadict['test'][:,coldict['loss']], 'b', label='Test loss')
	ax1.set_xlabel('Batch no.')
	ax1.set_ylabel('Loss')
	ax2.plot(batchno[1], datadict['test'][:,coldict['accuracy']], 'g', label='Test accuracy')
	ax2.plot(batchno[1], datadict['test'][:,coldict['iou']], 'c', label='Test IOU')
	ax2.set_ylabel('Accuracy/IOU')
	ax2.set_ylim(0.0,1.0)
	ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2, fontsize='small')
	ax2.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=1, fontsize='small')
	plt.savefig(graph_name + '_graph' + '.png')
	
def plot_label(np_arr, num_classes):
	plt.imshow(np_arr, cmap='jet', vmin=0, vmax=num_classes)
	plt.show()