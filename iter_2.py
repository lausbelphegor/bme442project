import numpy as np
import mne
from EEGModels import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

K.set_image_data_format('channels_last')

# Directory where the data is stored
data_dir = 'data'  # Change this to your actual data directory

def process_subject(subject_dir):
    edf_files = [f for f in os.listdir(subject_dir) if f.endswith('.edf')]
    X, y = [], []

    for edf_file in edf_files:
        edf_path = os.path.join(subject_dir, edf_file)
        raw = mne.io.read_raw_edf(edf_path, preload=True)
        raw.filter(2, None, method='iir')
        
        # Define event annotations
        events, event_id = mne.events_from_annotations(raw)
        tmin, tmax = 0., 2.  # Adjust these values based on your experiment protocol

        # Create epochs based on events
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False, baseline=None, preload=True)
        labels = epochs.events[:, -1]
        
        # Extract data and scale it
        X.append(epochs.get_data() * 1000)
        y.append(labels)
    
    return np.concatenate(X, axis=0), np.concatenate(y, axis=0)

def load_data(data_dir):
    subjects = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    X, y = [], []

    for subject in subjects:
        subject_dir = os.path.join(data_dir, subject)
        X_subject, y_subject = process_subject(subject_dir)
        X.append(X_subject)
        y.append(y_subject)
    
    return np.concatenate(X, axis=0), np.concatenate(y, axis=0)

# Load data for all subjects
X, y = load_data(data_dir)

# Define parameters
kernels, chans, samples = 1, X.shape[1], X.shape[2]

# Split data into train, validate, and test sets (adjust as needed)
train_size = int(0.5 * len(X))
validate_size = int(0.25 * len(X))

X_train = X[:train_size]
Y_train = y[:train_size]
X_validate = X[train_size:train_size + validate_size]
Y_validate = y[train_size:train_size + validate_size]
X_test = X[train_size + validate_size:]
Y_test = y[train_size + validate_size:]

# One-hot encode the labels
Y_train = np_utils.to_categorical(Y_train)
Y_validate = np_utils.to_categorical(Y_validate)
Y_test = np_utils.to_categorical(Y_test)

# Reshape data for EEGNet
X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
X_validate = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Initialize EEGNet model
model = EEGNet(nb_classes=Y_train.shape[1], Chans=chans, Samples=samples,
               dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16,
               dropoutType='Dropout')

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Set up model checkpointing
checkpointer = ModelCheckpoint(filepath='/checkpoints/checkpoint.h5', verbose=1, save_best_only=True)

# Train the model
fittedModel = model.fit(X_train, Y_train, batch_size=16, epochs=300, verbose=2,
                        validation_data=(X_validate, Y_validate),
                        callbacks=[checkpointer])

# Load best weights
model.load_weights('/checkpoints/checkpoint.h5')

# Evaluate model
probs = model.predict(X_test)
preds = probs.argmax(axis=-1)
acc = np.mean(preds == Y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))

# PyRiemann part (for comparison)
n_components = 2
clf = make_pipeline(XdawnCovariances(n_components), TangentSpace(metric='riemann'), LogisticRegression())
X_train_reshaped = X_train.reshape(X_train.shape[0], chans, samples)
X_test_reshaped = X_test.reshape(X_test.shape[0], chans, samples)
clf.fit(X_train_reshaped, Y_train.argmax(axis=-1))
preds_rg = clf.predict(X_test_reshaped)
acc2 = np.mean(preds_rg == Y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc2))

# Plot confusion matrices
names = ['rest', 'left fist', 'right fist', 'both fists', 'both feet']

cm_eegnet = confusion_matrix(Y_test.argmax(axis=-1), preds)
disp_eegnet = ConfusionMatrixDisplay(confusion_matrix=cm_eegnet, display_labels=names)
disp_eegnet.plot(cmap=plt.cm.Blues)
plt.title('EEGNet-8,2')
plt.show()

cm_rg = confusion_matrix(Y_test.argmax(axis=-1), preds_rg)
disp_rg = ConfusionMatrixDisplay(confusion_matrix=cm_rg, display_labels=names)
disp_rg.plot(cmap=plt.cm.Blues)
plt.title('xDAWN + RG')
plt.show()
