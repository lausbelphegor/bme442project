
# 2. METHOD

## Dataset Description

### EEGMMIDB Dataset
The EEG Motor Movement/Imagery Dataset (EEGMMIDB) from PhysioNet contains EEG recordings from multiple subjects performing various motor and imagery tasks. Each subject's EEG data is recorded using 64 electrodes placed according to the international 10-20 system. The dataset includes different conditions such as baseline (eyes open and closed) and task-specific movements (e.g., left fist, right fist, both fists, both feet).

### Data Preprocessing

- **Loading Data**: The EEG data for each subject is stored in EDF files, with each file representing a run of a specific task or baseline condition.
- **Filtering**: Each raw EEG recording is subjected to a powerline notch filter at 60 Hz to remove electrical noise, followed by a high-pass filter at 2 Hz to remove slow drifts.
- **Epoching**: The data is segmented into epochs. For baseline conditions, fixed-length epochs of 60 seconds are created. For task conditions, epochs of 4.1 seconds are created based on the event annotations in the data.
- **Normalization**: The EEG signals are converted from volts to microvolts for consistency.
- **Event Mapping**: Events are mapped to specific task labels, with each epoch assigned a label according to the task being performed during that epoch.

#### Example Preprocessing Code
```python
def process_run(edf_path, run_index, tmin=0, tmax=4.1):
    task_label = run_mapping.get(run_index, 'unknown')
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    raw.notch_filter(freqs=60.0)
    raw.filter(2, None, method='iir')
    
    events, event_id = mne.events_from_annotations(raw)
    labels = []
    epochs_data_list = []

    if task_label in ['baseline_open', 'baseline_closed']:
        epochs = mne.make_fixed_length_epochs(raw, duration=60, preload=True)
        epochs_data = epochs.get_data() * 1000
        labels = [event_id_mapping['T0']] * epochs_data.shape[0]
        if epochs_data.shape[0] > 0:
            epochs_data_list.append(epochs_data)
    else:
        epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, proj=False, baseline=(0,4.1), preload=True)
        epochs_data = epochs.get_data() * 1000
        for event in epochs.events:
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
        
        if epochs_data.shape[0] > 0:
            epochs_data_list.append(epochs_data)
    
    return epochs_data_list, labels, task_label
```

## Network Structure

The EEGNet model was used for this study. EEGNet is a compact convolutional neural network (CNN) designed specifically for EEG signal classification. It employs depthwise and separable convolutions to efficiently learn spatial and temporal features from EEG data.

### Model Architecture
The EEGNet model consists of the following layers:
- **Temporal Convolution**: Applies convolution across the time domain to learn temporal features.
- **Depthwise Convolution**: Applies convolution across the spatial domain for each temporal filter separately, learning spatial filters.
- **Separable Convolution**: Combines depthwise and pointwise convolutions to merge spatial filters.
- **Batch Normalization**: Normalizes the outputs of convolutional layers.
- **Activation Function**: Uses Exponential Linear Unit (ELU) activation for non-linearity.
- **Average Pooling**: Reduces the dimensionality of the data.
- **Fully Connected Layer**: Maps the learned features to the final output classes.

Below is the code snippet for the EEGNet model setup:
```python
model = NeuralNetClassifier(
    module=EEGNetv4,
    module__n_chans=64,
    module__n_outputs=5,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    optimizer__lr=0.1,
    iterator_train__shuffle=True,
    callbacks=[("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=200 - 1))],
    device='cuda' if cuda else 'cpu',
    verbose=True
)
```

### Implementation Details

**Training Details**
- **Training Data**: The preprocessed EEG data for each subject was split into training and testing sets using stratified k-fold cross-validation (5 folds).
- **Batch Size**: 64
- **Epochs**: Determined based on early stopping criteria.
- **Learning Rate**: 0.1, adjusted using cosine annealing.

**Validation Methods**
- **Stratified K-Fold Cross-Validation**: Ensured that each fold had a balanced representation of all classes.
- **Confusion Matrix**: Evaluated model performance on each fold by calculating sensitivity, specificity, positive predictive value (PPV), and negative predictive value (NPV) for each class.

Example validation code:
```python
from sklearn.metrics import accuracy_score, confusion_matrix

def process_fold(train_index, test_index):
    train_dataset = CustomDataset(X_combined[train_index], y_combined[train_index])
    test_dataset = CustomDataset(X_combined[test_index], y_combined[test_index])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model.fit(train_loader, y=None)  # Passing DataLoader directly

    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_batch_pred = model.predict(X_batch)
            y_pred.extend(y_batch_pred)
            y_true.extend(y_batch)
    
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    fold_sensitivities = []
    fold_specificities = []
    fold_ppvs = []
    fold_npvs = []
    
    for i in range(5):  # Assuming 5 classes
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        fold_sensitivities.append(sensitivity)
        fold_specificities.append(specificity)
        fold_ppvs.append(ppv)
        fold_npvs.append(npv)

    return fold_sensitivities, fold_specificities, fold_ppvs, fold_npvs

# Aggregating results from all folds
sensitivities = []
specificities = []
ppvs = []
npvs = []

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for train_index, test_index in skf.split(X_combined, y_combined):
        futures.append(executor.submit(process_fold, train_index, test_index))
    
    for future in concurrent.futures.as_completed(futures):
        fold_sensitivities, fold_specificities, fold_ppvs, fold_npvs = future.result()
        sensitivities.extend(fold_sensitivities)
        specificities.extend(fold_specificities)
        ppvs.extend(fold_ppvs)
        npvs.extend(fold_npvs)

# Calculate average metrics
avg_sensitivity = np.mean(sensitivities)
avg_specificity = np.mean(specificities)
avg_ppv = np.mean(ppvs)
avg_npv = np.mean(npvs)

print(f'Average Sensitivity: {avg_sensitivity}')
print(f'Average Specificity: {avg_specificity}')
print(f'Average PPV: {avg_ppv}')
print(f'Average NPV: {avg_npv}')
```
