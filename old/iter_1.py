import mne
import numpy as np
import tensorflow as tf
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import logging
import concurrent.futures
import os

# EEGNet implementation from EEGModels
from EEGModels import EEGNet

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SubjectData:
    def __init__(self, subject_id, task_runs, baseline_runs, data_path, n_samples):
        self.subject_id = subject_id
        self.task_runs = task_runs
        self.baseline_runs = baseline_runs
        self.data_path = data_path
        self.n_samples = n_samples
        self.task_data = {1: [], 2: [], 3: [], 4: []}
        self.task_labels = {1: [], 2: [], 3: [], 4: []}

    def load_data(self):
        logger.info(f'Processing subject {self.subject_id}')
        
        # Load baseline data
        baseline_data = []
        for baseline_run in self.baseline_runs:
            file_path = f'{self.data_path}/{self.subject_id}/S{self.subject_id}R{baseline_run:02d}.edf'
            if not os.path.exists(file_path):
                logger.warning(f'File {file_path} does not exist')
                continue
            raw = mne.io.read_raw_edf(file_path, preload=True, stim_channel='auto', verbose=False)
            raw.resample(sfreq=self.n_samples)
            baseline_data.append(raw.get_data())
        
        if not baseline_data:
            logger.warning(f'No baseline data for subject {self.subject_id}')
            return
        
        baseline_mean = np.mean(np.concatenate(baseline_data, axis=1), axis=1)

        for task, runs in self.task_runs.items():
            for run in runs:
                file_path = f'{self.data_path}/{self.subject_id}/S{self.subject_id}R{run:02d}.edf'
                if not os.path.exists(file_path):
                    logger.warning(f'File {file_path} does not exist')
                    continue
                raw = mne.io.read_raw_edf(file_path, preload=True, stim_channel='auto', verbose=False)
                raw.resample(sfreq=self.n_samples)
                events, event_id = mne.events_from_annotations(raw, verbose=False)
                if not events.size:
                    logger.warning(f'No events found in {file_path}')
                    continue
                epochs = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=2.5, baseline=None, preload=True, verbose=False)
                data = epochs.get_data()
                labels = epochs.events[:, -1]
                
                # Baseline correction
                baseline_corrected_data = data - baseline_mean[:, np.newaxis, np.newaxis]
                
                self.task_data[task].append(baseline_corrected_data)
                self.task_labels[task].append(labels)
        
        for task in self.task_data:
            if len(self.task_data[task]) > 0:
                self.task_data[task] = np.concatenate(self.task_data[task])
                self.task_labels[task] = np.concatenate(self.task_labels[task])
            else:
                self.task_data[task] = np.array([])
                self.task_labels[task] = np.array([])
        
        logger.info(f'Finished processing subject {self.subject_id}')

def load_all_data(subjects, task_runs, baseline_runs, data_path, n_samples):
    task_data = {1: [], 2: [], 3: [], 4: []}
    task_labels = {1: [], 2: [], 3: [], 4: []}
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        subject_data_objects = [SubjectData(f'S{subject:03d}', task_runs, baseline_runs, data_path, n_samples) for subject in subjects]
        futures = [executor.submit(subject_data.load_data) for subject_data in subject_data_objects]
        
        for future in concurrent.futures.as_completed(futures):
            pass

        for subject_data in subject_data_objects:
            for task in subject_data.task_data:
                if len(subject_data.task_data[task]) > 0:
                    task_data[task].append(subject_data.task_data[task])
                    task_labels[task].append(subject_data.task_labels[task])
                else:
                    logger.warning(f'No data for task {task} in subject {subject_data.subject_id}')
    
    for task in task_data:
        if len(task_data[task]) > 0:
            task_data[task] = np.concatenate(task_data[task])
            task_labels[task] = np.concatenate(task_labels[task])
        else:
            task_data[task] = np.array([])
            task_labels[task] = np.array([])
            logger.warning(f'No data for task {task} in any subject')

    return task_data, task_labels

# Define subjects and runs
subjects = range(1, 110)
# Runs for different tasks:
task_runs = {
    1: [3, 7, 11],  # Task 1: open and close left or right fist
    2: [4, 8, 12],  # Task 2: imagine opening and closing left or right fist
    3: [5, 9, 13],  # Task 3: open and close both fists or both feet
    4: [6, 10, 14]  # Task 4: imagine opening and closing both fists or both feet
}
# Baseline runs: eyes open (1), eyes closed (2)
baseline_runs = [1, 2]

# Path to the dataset
data_path = 'data'

# Number of samples per epoch (resampling frequency)
n_samples = 160  # Sampling frequency

# Load data
task_data, task_labels = load_all_data(subjects, task_runs, baseline_runs, data_path, n_samples)

# Combine all tasks into one dataset, skipping empty arrays
X = []
y = []
for task in task_data:
    if task_data[task].size > 0:
        X.append(task_data[task])
        y.append(task_labels[task])

# Add a check to ensure there is data before attempting to concatenate
if X and y:
    X = np.concatenate(X)
    y = np.concatenate(y)
else:
    raise ValueError('No valid data available for concatenation')

# Map the labels to new labels: 0 - Left Fist, 1 - Right Fist, 2 - Both Fists, 3 - Both Feet
label_map = {1: 0, 2: 1, 3: 2, 4: 3}
y = np.vectorize(label_map.get)(y)

# Preprocess data
X = np.expand_dims(X, axis=-1)

# Check the distribution of labels
unique, counts = np.unique(y, return_counts=True)
label_distribution = dict(zip(unique, counts))
logger.info(f'Label distribution: {label_distribution}')

# Ensure there are exactly 4 unique classes in the labels
unique_classes = np.unique(y)
logger.info(f'Unique classes in the labels: {unique_classes}')

if len(unique_classes) != 4:
    raise ValueError(f'Expected 4 unique classes, but found {len(unique_classes)}')

# Convert labels to categorical
y = np_utils.to_categorical(y, num_classes=4)

# Check the number of samples per epoch
samples_per_epoch = X.shape[2]
logger.info(f'Number of samples per epoch: {samples_per_epoch}')

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up TensorFlow to use the GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        logger.info('GPU is available and will be used for training.')
    except RuntimeError as e:
        logger.error(e)
else:
    logger.warning('GPU is not available, training will use CPU.')

# EEGNet model
model = EEGNet(nb_classes=4, Chans=64, Samples=samples_per_epoch, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25, dropoutType='Dropout')

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint(filepath='eegnet_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')
callbacks = [checkpoint]

# Train the model
history = model.fit(X_train, y_train, batch_size=16, epochs=100, validation_data=(X_test, y_test), callbacks=callbacks, verbose=2)

# Load the best model
best_model = load_model('eegnet_model.h5')

# Evaluate the model
score = best_model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {score[1]:.4f}')
