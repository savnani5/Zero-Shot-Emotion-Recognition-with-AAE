import numpy as np
import deepdish as dd
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from scipy.io import savemat

dat = dd.io.load('featuresABHI.h5')
lab = dd.io.load('labelsABHI.h5')
##############################
labels = {}
features = {}
##############################
lis = []
for k,v in lab.items():
	lis.append(v)
##############################
lis3= []
for i,num in enumerate(lis):
	if num==3:
		lis3.append(i)
lis6= []
for j,num6 in enumerate(lis):
	if num6==6:
		lis6.append(j)
list_unseen = lis3+lis6
unseen = list_unseen
list_unseen = np.asarray(list_unseen).reshape((len(list_unseen),1))
list_unseen = list_unseen + 1
print(list_unseen.shape,"Unseen")
##############################
model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
emotions = ['joy','relief','pride','shame','anger','surprise','amusement','sadness','fear','neutral','disgust']
attrib = []
for em in emotions:
	vector = model[em]
	attrib.append(vector)
attrib = np.asarray(attrib)
attrib = attrib.T
print(attrib.shape,"Att")
##############################
list_seen= []
for k,nums in enumerate(lis[350:650]):
	if nums != 3:
		if nums==3:
			print("Ho gaya")
		list_seen.append((k+350,nums))
test_seen = []
for numl in list_seen:
	if numl[1] != 6:
		test_seen.append(numl[0])
seen = test_seen
test_seen = np.asarray(test_seen).reshape((len(test_seen),1))
test_seen = test_seen + 1
print(test_seen.shape,"Seen")
##############################
elim = seen + unseen
lis_id = [item for item in range(1,1443)]
train_val = [z for z in lis_id if z not in elim]
train_val = np.asarray(train_val).reshape((len(train_val),1))
print(train_val.shape,"Trainval")
##############################
labels['test_unseen_loc'] = list_unseen
labels['test_seen_loc'] = test_seen
labels['att'] = attrib
labels['trainval_loc'] = train_val
##############################
###########FEATURES###########
##############################
label_list = np.asarray(lis).reshape(len(lis),1)
##############################
feat = dd.io.load('output.h5')
featr = []
for kf,vf in feat.items():
	featr.append(vf)
featr = np.asarray(featr)
featr = featr.T
print(featr.shape,"Features")
##############################
features['labels'] = label_list
features['features'] = featr
##############################
savemat("featuresT.mat", features)
savemat("labelsT.mat", labels)
# dd.io.save('featuresTEST.h5',features)
# dd.io.save('labelsTEST.h5',labels)
