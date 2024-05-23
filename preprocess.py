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
    'T0': 1,  # Rest
    'T1_left': 2,  # Left fist
    'T1_both': 3,  # Both fists
    'T2_right': 4,  # Right fist
    'T2_both': 5  # Both feet
}

def process_run(edf_path, run_index, tmin=0, tmax=4.1):
    task_label = run_mapping.get(run_index, 'unknown')
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    raw.notch_filter(freqs=60.0)  # powerline notch filter
    raw.filter(2, None, method='iir')  # High-pass filter at 2 Hz
    
    # Define event annotations
    events, event_id = mne.events_from_annotations(raw)
    labels = []
    epochs_data_list = []

    # Handle baselines differently since they are long T0 periods
    if task_label in ['baseline_open', 'baseline_closed']:
        tmin_baseline = 0  # start from the beginning
        tmax_baseline = 60  # 60 seconds for baseline
        epochs = mne.make_fixed_length_epochs(raw, duration=tmax_baseline, preload=True, overlap=0)
        epochs_data = epochs.get_data() * 1000  # Convert to microvolts

        labels = [event_id_mapping['T0']] * epochs_data.shape[0]  # Label all as T0
        if epochs_data.shape[0] > 0:  # Only append if there are valid epochs
            epochs_data_list.append(epochs_data)
    else:
        # Create epochs based on static tmin and tmax values
        epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, proj=False, baseline=(0,4.1), preload=True)
        epochs_data = epochs.get_data() * 1000  # Convert to microvolts

        for i, event in enumerate(epochs.events):
            if task_label in ['task_1', 'task_2']:
                if event[2] == event_id.get('T0'):
                    labels.append(event_id_mapping['T0'])
                elif event[2] == event_id.get('T1'):
                    labels.append(event_id_mapping['T1_left'])
                elif event[2] == event_id.get('T2'):
                    labels.append(event_id_mapping['T2_right'])
            elif task_label in ['task_3', 'task_4']:
                if event[2] == event_id.get('T0'):
                    labels.append(event_id_mapping['T0'])
                elif event[2] == event_id.get('T1'):
                    labels.append(event_id_mapping['T1_both'])
                elif event[2] == event_id.get('T2'):
                    labels.append(event_id_mapping['T2_both'])
            else:
                labels.append(event_id_mapping['T0'])
            print(labels)
        
        if epochs_data.shape[0] > 0:  # Only append if there are valid epochs
            epochs_data_list.append(epochs_data)
    
    return epochs_data_list, labels, task_label

def pad_or_trim(data, target_length):
    def pad(array):
        if array.shape[2] < target_length:
            pad_width = target_length - array.shape[2]
            return np.pad(array, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
        return array[:, :, :target_length]
    
    return np.concatenate([pad(d) for d in data], axis=0)

def save_data(data_dir, tmin=0, tmax=4.1):
    subjects = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    max_samples = int((tmax - tmin) * 160)  # Assuming a sampling rate of 160 Hz
    baseline_samples = 60 * 160  # Assuming 60 seconds for baselines

    for subject in subjects:
        subject_dir = os.path.join(data_dir, subject)
        edf_files = [f for f in os.listdir(subject_dir) if f.endswith('.edf')]
        
        for edf_file in edf_files:
            edf_path = os.path.join(subject_dir, edf_file)
            run_index = int(edf_file.split('R')[1].split('.')[0])  # Extract run index from filename
            epochs_data_list, labels, task_label = process_run(edf_path, run_index, tmin, tmax)
            
            if len(labels) > 0 and epochs_data_list:  # Only process if there are labels and epochs
                if task_label in ['baseline_open', 'baseline_closed']:
                    padded_data = pad_or_trim(epochs_data_list, baseline_samples)
                else:
                    padded_data = pad_or_trim(epochs_data_list, max_samples)
                
                labels = np.array(labels)

                # Create the save directory
                save_dir = os.path.join('preprocessed_data', subject, task_label)
                os.makedirs(save_dir, exist_ok=True)

                # Save the data for each run
                save_path = os.path.join(save_dir, f"run_{run_index}.pkl")
                with open(save_path, 'wb') as f:
                    pickle.dump({'X': padded_data, 'y': labels}, f)
                print(f"Saved {subject} - {task_label} - run {run_index} data.")

# Save preprocessed data for all subjects
save_data(data_dir)

print("All preprocessed data saved.")
