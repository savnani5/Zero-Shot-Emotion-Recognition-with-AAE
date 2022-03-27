import csv
import h5py
import os
import numpy as np
from gensim.models import KeyedVectors
# filename = "features4DCVAEGCN.h5"

# with h5py.File(filename, "r") as f:
#     print("Keys: %s" % f.keys())
#     a_group_key = list(f.keys())[0]
#     data = list(f[a_group_key])
savepath ="./"

# filename = 'features4DCVAEGCN.h5'
# filename = 'test.h5'
# f = h5py.File(filename, 'r')
# for idx in range(len(f.keys())):
#     a_group_key = list(f.keys())[idx]
#     # print(a_group_key)
#     data = np.array(f[a_group_key])
#     # print(len(data))
#     # if idx==0:
#     #     print(len(data))
#         # Get the data
#     np.savetxt(os.path.join(savepath, a_group_key+'.csv'), data, delimiter=',')

labels = 'labelsABHI.h5'
l = h5py.File(labels, 'r')
for idl in range(len(l.keys())):
	print(l[list(l.keys())[idl]][()])

# model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
# emotions = ['joy','relief','pride','shame','anger','surprise','amusement','sadness','fear','neutral','disgust']
# for em in emotions:
# 	vector = model[em]
# 	print(vector.shape)