import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
plt.switch_backend('agg') #stops DISPLAY error when using SSH

def make_graph(logfile, graph_path):
	df = pd.read_csv(logfile, header=0)
	data = df.values
	coldict = {column:idx for idx, column in enumerate(df.columns.tolist())}
	datadict = {'train': data[data[:,-1]==0], 'test': data[data[:,-1]==1]}
	batches_per_epoch = np.max(datadict['train'][:,coldict['batch']])
	get_batch  = lambda set_: datadict[set_][:,coldict['batch']] \
							+ datadict[set_][:,coldict['epoch']] \
							* batches_per_epoch
	batchno = map(get_batch, ['train', 'test'])
	
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.plot(batchno[0], datadict['train'][:,coldict['loss']], 'r--', \
			 batchno[1], datadict['test'][:,coldict['loss']], 'b--')
	ax1.set_xlabel('batch no.')
	ax1.set_ylabel('loss')
	ax2.plot(batchno[1], datadict['test'][:,coldict['accuracy']], 'g--')
	ax2.set_ylabel('accuracy')
	ax2.set_ylim(0,100)
	plt.savefig(graph_path)

def plot_label(np_arr, num_classes):
	plt.imshow(np_arr, cmap='jet', vmin=0, vmax=num_classes)
	plt.show()