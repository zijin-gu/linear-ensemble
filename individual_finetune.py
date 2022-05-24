import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from usefuncs import *
from dataloaders import NSDdataset
from models import *
import argparse
from itertools import repeat
import warnings
warnings.filterwarnings("ignore")
from collections import OrderedDict

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Encoding model')
	parser.add_argument('--subject', default=8, type=int, help='Subject ID')
	parser.add_argument('--roi', default='V1v', type=str, help='Roi name')
	parser.add_argument('--level', default='ROI', type=str, help='ROI level or VOXEL level')
	parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
	parser.add_argument('--readout_type', default='linear', type=str, help='Readout type')
	parser.add_argument('--method', default='finetune', type=str, help='Fixed/finetune/scratch')
	parser.add_argument('--train_size', default=None, type=int, help='Train size')
	parser.add_argument('--epoch', default=100, type=int, help='Num epoches')
	parser.add_argument('--retrain', default=False, type=bool, help='Build on NSD or not')

	args = parser.parse_args()

	subject = args.subject
	roi = args.roi
	method = args.method
	readout_type = args.readout_type
	batch_size = args.batch_size
	level = args.level
	train_size = args.train_size
	if args.retrain:
		model_dir = f'./ckpt_{level}/nsd_finetune/S{str(subject)}/'       
	else:        
		model_dir = f'./ckpt_{level}/S{str(subject)}/'
	if train_size:
		model_dir = model_dir + f'size{str(train_size)}/'
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)
	model_base = 'resnet50_%s_%s_%s' % (roi, method, readout_type)

	params = {'batch_size': batch_size,'shuffle': True,'num_workers': 6}
	
	# Generators
	training_set = NSDdataset(mode='train', subject=subject, roi=roi, train_size=train_size)
	training_generator = torch.utils.data.DataLoader(training_set,  **params)

	validation_set = NSDdataset(mode='val', subject=subject, roi=roi, train_size=None)
	validation_generator = torch.utils.data.DataLoader(validation_set,  **params)

	n_neurons = training_set.responses.shape[1]
	print('Number of neurons', n_neurons)
   
	if method == 'finetune':
		finetune, pretrained = True, True
	elif method == 'scratch':
		finetune, pretrained = True, False
	else:
		finetune, pretrained = False, True

	core = FeatCore(pretrained=pretrained, finetune=finetune)
# 	if readout_type =='factorized':
# 		readout = SpatialXFeatureLinear(core(torch.randn(1, 3, 256, 256)).size()[1:], n_neurons,  bias=True)
	if readout_type == 'linear':
		readout = SimpleLinear(core(torch.randn(1, 3, 256, 256)).size()[1:], n_neurons)        

	model = Encoder(core, readout)
	model.cuda()
	
	### Define training parameters
	schedule = [1e-4]
	criterion = masked_MSEloss 
	best_corr = -1
	patience = 20
	iter_tracker = 0 
	accumulate_gradient=2
	n_epochs = args.epoch
	
	### Start training 
	for opt, lr in zip(repeat(torch.optim.Adam), schedule):
		print('Training with learning rate', lr)
		optimizer = opt(model.parameters(), lr=lr)
		optimizer.zero_grad()
		iteration = 0
		if args.retrain:
			nsd_params = np.load('./ckpt_ROI/nsd_8subj_readout_params.npy', allow_pickle=True).tolist()
			weight = np.delete(nsd_params['weight'], subject-1, 0)
			bias = np.delete(nsd_params['bias'], subject-1, 0)
			mean_weights = torch.mean(torch.from_numpy(weight), axis=0)
			mean_bias = torch.mean(torch.from_numpy(bias), axis=0)
			state_dict = OrderedDict()
			state_dict['linear.weight'] = mean_weights
			state_dict['linear.bias'] = mean_bias
			model.readout.load_state_dict(state_dict)

		assert accumulate_gradient > 0, 'accumulate_gradient needs to be > 0'
		for epoch in range(n_epochs):
			for x_batch, y_batch in training_generator:
				obj = full_objective(model, x_batch.cuda().float(), y_batch.cuda().float(), criterion)
				obj.backward()
				iteration += 1 
				if iteration % accumulate_gradient == accumulate_gradient - 1:
					optimizer.step()
					optimizer.zero_grad()
				if iteration % 100 == 0:
					model.eval()
					true, preds = compute_predictions(validation_generator, model)
					val_corr = compute_scores(true, preds)
					print('Val correlation:', val_corr)
					model.train()
					is_best = val_corr >= best_corr
					if is_best:
						best_corr = val_corr.copy()
						iter_tracker = 0        
						save_checkpoint({'epoch': epoch + 1,
											'state_dict': model.state_dict(),
											'optim_dict': optimizer.state_dict()},
											is_best = is_best,
											checkpoint=model_dir, model_str=model_base)
					else:
						iter_tracker += 1
						if iter_tracker == patience: 
							print('Training complete')
							break 
			if iter_tracker == patience: 
							print('Training complete')
							break  
