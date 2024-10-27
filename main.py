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
    learning_rate = 0.001, momentum=0.0, nesterov=False, name='SGD'
)

model = Sequential([
    Dense(4, activation='relu', input_dim=4),
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
        voltage = np.array(file['data'][[0], 1:])

        voltage = voltage.astype(float)
        Parr = np.full((1, voltage.shape[1]), Pconc/10)
        Narr = np.full((1, voltage.shape[1]), Nconc/10)
        OSarr = np.full((1, voltage.shape[1]), offset/400)

        training_set_inputs = np.concatenate((OSarr, Parr, Narr, voltage), axis=0).T
        #For phase response
        training_set_outputs = np.array(file['data'][[3], 1:]).T
        #For optical loss: training_set_outputs = np.array(file['data'][[4], 1:]).T

    return(training_set_inputs, training_set_outputs)

#Change path here
path = "#insert path directing to the file containing all offset files"
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
        model.fit(training_set_inputs, training_set_outputs, epochs=2, verbose=1, batch_size=16, shuffle=True)


#This is for inputting values individually
decision = "y"
while (decision=="y"):
    OffSet = float(input("Offset: "))
    Pconcentration = float(input("Pconcentration: "))
    Nconcentration = float(input("Nconcentration: "))
    Volts = float(input("Volts: "))
    test_input = [OffSet, Pconcentration, Nconcentration, Volts]
    test_input = np.array([OffSet/400, Pconcentration/10, Nconcentration/10, Volts])
    test_input = test_input.reshape(1,4)
    print(model.predict(test_input))

    decision = str(input("continue? y/n"))


#This is for testing the error of a file. First input will ask for file path, second input will ask for offset. 
#If you enter "end" for filepath input, while loop stops
# testpath = input()
# while (testpath!="end"):
#     offsetvalue = int(input())
#     test = glob.glob(os.path.join(testpath, "*.mat")) 
#     error=0
#     for f in mat_files: 
#         test_input, test_output = getdata(f, offsetvalue)
#         for i in range (0, 30, 5):
#             x = test_input[i].reshape(1,4)
#             error += abs(float(model.predict(x))-float(test_output[i]))
#         print(error)
#     print(error/486) 
#     testpath = input()
