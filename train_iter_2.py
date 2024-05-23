import numpy as np
import pickle
import mne
from models.EEGModels import EEGNet

import torch
from braindecode.models import EEGNetv4, EEGConformer, ATCNet, EEGITNet, EEGInception
from skorch import NeuralNetClassifier
from models import cuda
from skorch.callbacks import LRScheduler, TrainEndCheckpoint

from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K


from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import os
import visualkeras

# Set image data format
K.set_image_data_format('channels_last')

import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

# Define the base directory
base_dir = './preprocessed_data/'

# Initialize lists to hold the combined data
X_combined = []
y_combined = []

# with open('C:\\temp\\BME_442\\bme442project\\preprocessed_data\S001\\task_1\\run_3.pkl', 'rb') as f:
#     data = pickle.load(f)
#     print(data['y'])

# Load data from each subject
for subject in os.listdir(base_dir):
    subject_dir = os.path.join(base_dir, subject)
    if os.path.isdir(subject_dir):
        for task in ['task_1', 'task_2', 'task_3', 'task_4']:
            task_dir = os.path.join(subject_dir, task)
            if os.path.isdir(task_dir):
                for run_file in os.listdir(task_dir):
                    if run_file.endswith('.pkl'):
                        run_path = os.path.join(task_dir, run_file)
                        print(run_path)
                        with open(run_path, 'rb') as f:
                            data = pickle.load(f)
                            X_combined.append(data['X'])
                            y_combined.append(data['y'])
                            # print(X_combined)

# Convert lists to arrays
X_combined = np.concatenate(X_combined, axis=0)
y_combined = np.concatenate(y_combined, axis=0)
P
# Check the shapes of the combined datasets
print("X_combined shape:", X_combined.shape)
print("y_combined shape:", y_combined.shape)

unique, counts = np.unique(y_combined, return_counts=True)

# Ensure that the number of samples is correct for y
if X_combined.shape[0] != y_combined.shape[0]:
    raise ValueError("Mismatch between number of samples in X_combined and y_combined.")

# Split data into train, validate, and test sets (adjust as needed)
train_size = int(0.5 * len(X_combined))
validate_size = int(0.25 * len(X_combined))

X_train = X_combined[:train_size]
Y_train = y_combined[:train_size]
X_validate = X_combined[train_size:train_size + validate_size]
Y_validate = y_combined[train_size:train_size + validate_size]
X_test = X_combined[train_size + validate_size:]
Y_test = y_combined[train_size + validate_size:]

# Check the shapes of the splits before one-hot encoding
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_validate shape:", X_validate.shape)
print("Y_validate shape:", Y_validate.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

# One-hot encode the labels
Y_train = np_utils.to_categorical(Y_train)
Y_validate = np_utils.to_categorical(Y_validate)
Y_test = np_utils.to_categorical(Y_test)

# Reshape data for EEGNet
kernels, chans, samples = 1, X_train.shape[1], X_train.shape[2]

X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
X_validate = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

print('Y_train shape', Y_train.shape)
print('Y_test shape', Y_test.shape)

# Initialize EEGNet model
model = EEGNet(nb_classes=Y_train.shape[1], Chans=chans, Samples=samples,
               dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16,
               dropoutType='Dropout')

visualkeras.layered_view(model, to_file='model.png').show()

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Set up model checkpointing
checkpointer = ModelCheckpoint(filepath='./tmp/checkpoint.h5', verbose=1, save_best_only=True)

# Train the model
fittedModel = model.fit(X_train, Y_train, batch_size=16, epochs=300, verbose=2,
                        validation_data=(X_validate, Y_validate),
                        callbacks=[checkpointer])

# Load best weights
model.load_weights('./tmp/checkpoint.h5')

# Print checkpoint folder full path
print("Model checkpoint folder:", os.path.abspath('./tmp/checkpoint.h5'))

# Evaluate model
probs = model.predict(X_test)
preds = probs.argmax(axis=-1)
acc = np.mean(preds == Y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))

# Plot confusion matrices
names = ['rest', 'left fist', 'right fist', 'both fists', 'both feet']

cm_eegnet = confusion_matrix(Y_test.argmax(axis=-1), preds)
disp_eegnet = ConfusionMatrixDisplay(confusion_matrix=cm_eegnet, display_labels=names)
disp_eegnet.plot(cmap=plt.cm.Blues)
plt.title('EEGNet-8,2')
plt.show()
