from sklearn.linear_model import LinearRegression
import argparse
import numpy as np
from scipy.stats import pearsonr
from dataloaders import NSDdataset
from models import *
from dataloaders import *
import os
import random
import torch
parser = argparse.ArgumentParser(description='Train and test new data')
parser.add_argument('--subject', default=2, type=int, help='Subject ID: 2 to 7')
parser.add_argument('--roi', default='FFA1', type=str, help='ROI name')
parser.add_argument('--train_size', default=None, type=int, help='Number of samples to fit')
args = parser.parse_args()
subject = args.subject
roi = args.roi
train_size = args.train_size
batch_size = 50
level = 'ROI'
method = 'finetune'
readout_type = 'linear'
pretrained, finetune = True, True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# params = {'batch_size': batch_size,'shuffle': False}
# training_set = NSDdataset(mode='train', subject=subject, roi=roi, train_size=train_size)
# training_generator = torch.utils.data.DataLoader(training_set,  **params)
# test_set = NSDdataset(mode='test', subject=subject, roi=roi, train_size=None)
# test_generator = torch.utils.data.DataLoader(test_set,  **params)

# train_act, train_true = np.zeros([len(training_set), 7]), np.zeros([len(training_set), 1])
# test_act, test_true = np.zeros([len(test_set), 7]), np.zeros([len(test_set), 1])

# train_subj = list(set(range(1,9)) - set([subject]))
# for i, nsd_subject in enumerate(train_subj):
#     print(nsd_subject)
#     model_dir = f'./ckpt_{level}/S{str(nsd_subject)}/'
#     model_base = 'resnet50_%s_%s_%s' % (roi, method, readout_type)
#     core = FeatCore(pretrained=pretrained, finetune=finetune)
#     if readout_type == 'linear':
#         readout = SimpleLinear(core(torch.randn(1, 3, 256, 256)).size()[1:], 1)
#     predictor = Encoder(core, readout)
#     predictor.to(device)
#     restore_file = 'best_' + model_base
#     restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
#     checkpoint = torch.load(restore_path, map_location=device)
#     state_dict = checkpoint['state_dict']
#     predictor.load_state_dict(state_dict, strict=False)
#     predictor.eval()
    
#     pred = []
#     true = []
#     for (img, act) in training_generator:
#         pred.append(predictor(img.to(device)).detach().cpu().numpy())
#         true.append(act.numpy())
#     train_act[:,i] = np.vstack(pred).reshape(-1)
#     train_true[:,0] = np.vstack(true).reshape(-1)
    
#     pred = []
#     true = []
#     for (img, act) in test_generator:
#         pred.append(predictor(img.to(device)).detach().cpu().numpy())
#         true.append(act.numpy())
#     test_act[:,i] = np.vstack(pred).reshape(-1)
#     test_true[:,0] = np.vstack(true).reshape(-1)

# np.save(f'./output/nsd_ensemble/nsd_pred_responses/S{subject}_{roi}_train.npy', train_act)
# np.save(f'./output/nsd_ensemble/nsd_pred_responses/S{subject}_{roi}_test.npy', test_act)

nsddata_dir = '/home/zg243/nsd/LE/data/nsddata/'

train_act = np.load(f'./output/nsd_ensemble/nsd_pred_responses/S{subject}_{roi}_train.npy')
test_act = np.load(f'./output/nsd_ensemble/nsd_pred_responses/S{subject}_{roi}_test.npy')
train_true = np.load(nsddata_dir + f'S{subject}_train_responses.npy', allow_pickle=True).tolist()[roi]
test_true = np.load(nsddata_dir + f'S{subject}_test_responses.npy', allow_pickle=True).tolist()[roi]

le_pred = np.zeros([100, len(test_true)])
le_params = np.zeros([100, 8])
le_acc = np.zeros(100)
for repeat in range(100):
    print('repeat %d'%repeat)
    random.seed(repeat)
    indices = random.sample(range(len(train_true)), train_size)
    X = train_act[indices]
    y = train_true[indices]
    X_test = test_act
    y_test = test_true
    reg = LinearRegression().fit(X, y)
    params = np.append(reg.coef_, reg.intercept_)
    pred = reg.predict(X_test)
    acc = pearsonr(pred.reshape(-1), y_test.reshape(-1))[0]
    #mean_pred = np.mean(X_test, axis=1)
    #mean_acc = pearsonr(mean_pred, y_test.reshape(-1))[0]
    
    le_pred[repeat] = pred.reshape(-1)
    le_params[repeat] = params
    le_acc[repeat] = acc
output = {'le_pred': le_pred, 'le_params': le_params, 'le_acc': le_acc}
result_dir = './output/nsd_ensemble/repeat100/size%d/'%train_size
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
np.save(result_dir + 'S%d_%s.npy'%(subject, roi), output)

