# import os 
# import glob 
# import h5py
# import numpy as np

# import tensorflow as tf
# from tensorflow import keras
# from keras.models import Sequential # type: ignore
# from keras.layers import Dense, Dropout, Normalization # type: ignore
# from keras.optimizers import SGD # type: ignore
# from keras.losses import MeanSquaredError as MSE # type: ignore

# tf.keras.optimizers.SGD(
#     learning_rate = 0.005, momentum=0.01, nesterov=False, name='SGD'
# )

# model = Sequential([
#     Dense(3, activation='relu', input_dim=3),
#     Dropout(0.2),
#     Dense(14, activation='relu'),
#     Dense(1)
# ])

# model.compile(
#     optimizer=SGD(), loss=MSE()
# )

# def getdata(f, offset):
#     with h5py.File(f, 'r') as file:
#         print(f)
#         conc = f.split("Concentration")[-1]
#         Pconc = int(conc.split("_")[0])
#         split_mat = conc.split("_")[-1]
#         Nconc = int(split_mat.split(".mat")[0])

#         idata = np.array(file['data'][[0], 1:])
#         idata = idata.astype(float)
#         Parr = np.full((1, idata.shape[1]), Pconc)
#         Narr = np.full((1, idata.shape[1]), Nconc)
#         OSarr = np.full((1, idata.shape[1]), offset)

#         training_set_inputs = np.concatenate((OSarr, Parr, Narr, idata), axis=0).T
#         training_set_outputs = np.array(file['data'][[3], 1:]).T

#     return(training_set_inputs, training_set_outputs)

# path = "/Users/phyu34199/Desktop/Astar/alloffsets/testingfolder"
# folders = glob.glob(os.path.join(path, "*nm")) 
# #mat_files = glob.glob(os.path.join(path, "*.mat")) 

# for fold in folders: 
#     mat_files = glob.glob(os.path.join(fold, "*.mat")) 
#     for file in mat_files:
#         offsetline = file.split("offset_")[1]
#         offsetline = offsetline.split("nm")[0]
#         offset = int(offsetline.split("_")[-1])
#         if (offsetline[0] == 'n'):
#             offset = -offset
#         training_set_inputs, training_set_outputs = getdata(file, offset)
#         print(training_set_inputs, "\n", training_set_outputs)
        



import os 
import glob 
import h5py
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential # type: ignore
from keras.layers import Dense, Dropout, Normalization # type: ignore
from keras.optimizers import SGD # type: ignore
from keras.losses import MeanSquaredError as MSE # type: ignore

tf.keras.optimizers.SGD(
    learning_rate = 0.001, momentum=0.01, nesterov=False, name='SGD'
)

model = Sequential([
    Dense(4, activation='relu', input_dim=4),
    #Dropout(0.2),
    Dense(14, activation='relu'),
    Dense(1)
])

model.compile(
    optimizer=SGD(), loss=MSE()
)

def getdata(f, offset):
    with h5py.File(f, 'r') as file:
        conc = f.split("Concentration")[-1]
        Pconc = int(conc.split("_")[0])
        split_mat = conc.split("_")[-1]
        Nconc = int(split_mat.split(".mat")[0])
        idata = np.array(file['data'][[0], 1:])

        idata = idata.astype(float)
        Parr = np.full((1, idata.shape[1]), Pconc/10)
        Narr = np.full((1, idata.shape[1]), Nconc/10)
        OSarr = np.full((1, idata.shape[1]), offset/400)

        training_set_inputs = np.concatenate((OSarr, Parr, Narr, idata), axis=0).T
        training_set_outputs = np.array(file['data'][[3], 1:]).T

    return(training_set_inputs, training_set_outputs)

path = "/Users/phyu34199/Desktop/Astar/alloffsets/testingfolder"
folders = glob.glob(os.path.join(path, "*nm")) 

for fold in folders: 
    mat_files = glob.glob(os.path.join(fold, "*.mat")) 
    for file in mat_files:
        offsetline = file.split("offset_")[1]
        offsetline = offsetline.split("nm")[0]
        offset = float(offsetline.split("_")[-1])
        if (offsetline[0] == 'n'):
            offset = -offset
        training_set_inputs, training_set_outputs = getdata(file, offset)
        #model.fit(training_set_inputs, training_set_outputs, epochs=2, verbose=1, batch_size=16, shuffle=True)
        print(training_set_inputs, "\n", training_set_outputs)
