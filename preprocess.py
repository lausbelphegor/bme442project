import os
import numpy as np
import mne
import pickle

# Directory where the data is stored
data_dir = 'data'  # Change this to your actual data directory

# Mapping of run indices to tasks/baselines
run_mapping = {
    1: 'baseline_open',
    2: 'baseline_closed',
    3: 'task_1',
    4: 'task_2',
    5: 'task_3',
    6: 'task_4',
    7: 'task_1',
    8: 'task_2',
    9: 'task_3',
    10: 'task_4',
    11: 'task_1',
    12: 'task_2',
    13: 'task_3',
    14: 'task_4'
}

# Mapping of event IDs to labels
event_id_mapping = {
    'T0': 0,  # Rest
    'T1_left': 1,  # Left fist
    'T1_both': 2,  # Both fists
    'T2_right': 3,  # Right fist
    'T2_both': 4  # Both feet
}

def process_subject(subject_dir):
    edf_files = [f for f in os.listdir(subject_dir) if f.endswith('.edf')]
    X, y = [], []
    max_samples = 0

    for edf_file in edf_files:
        edf_path = os.path.join(subject_dir, edf_file)
        run_index = int(edf_file.split('R')[1].split('.')[0])  # Extract run index from filename
        task_label = run_mapping.get(run_index, 'unknown')

        raw = mne.io.read_raw_edf(edf_path, preload=True)
        raw.filter(2, None, method='iir')  # High-pass filter at 2 Hz
        
        # Define event annotations
        events, event_id = mne.events_from_annotations(raw)
        tmin, tmax = 0., 0.003  # Adjust these values based on your experiment protocol

        # Create epochs based on events
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False, baseline=None, preload=True)
        labels = []

        for e in epochs.events:
            if task_label in ['task_1', 'task_2']:
                if e[2] == event_id['T1']:
                    labels.append(event_id_mapping['T1_left'])
                elif e[2] == event_id['T2']:
                    labels.append(event_id_mapping['T2_right'])
            elif task_label in ['task_3', 'task_4']:
                if e[2] == event_id['T1']:
                    labels.append(event_id_mapping['T1_both'])
                elif e[2] == event_id['T2']:
                    labels.append(event_id_mapping['T2_both'])
            else:
                labels.append(event_id_mapping['T0'])
        
        if len(labels) > 0:  # Only append if there are labels
            max_samples = max(max_samples, epochs.get_data().shape[2])
            X.append(epochs.get_data() * 1000)
            y.append(labels)
    
    return X, y, max_samples

def pad_or_trim(data, target_length):
    def pad(array):
        if array.shape[2] < target_length:
            pad_width = target_length - array.shape[2]
            return np.pad(array, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
        return array[:, :, :target_length]
    
    return np.concatenate([pad(d) for d in data], axis=0)

def load_data(data_dir):
    subjects = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    X, y = [], []
    max_samples = 0

    for subject in subjects:
        subject_dir = os.path.join(data_dir, subject)
        X_subject, y_subject, subject_max_samples = process_subject(subject_dir)
        if len(y_subject) > 0:  # Only append if there are labels
            max_samples = max(max_samples, subject_max_samples)
            X.extend(X_subject)  # Use extend to add the elements of the list
            y.extend(y_subject)  # Use extend to add the elements of the list

    # Pad or trim all data to the maximum number of samples
    X = pad_or_trim(X, max_samples)
    y = np.array(y)
    
    return X, y

# Load and preprocess data for all subjects
X, y = load_data(data_dir)

# Save the preprocessed data
with open('preprocessed_data.pkl', 'wb') as f:
    pickle.dump((X, y), f)

print("Preprocessed data saved.")
