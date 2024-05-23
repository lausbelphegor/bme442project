import matplotlib.pyplot as plt
import mne
import os


# Load the EDF file
file_path = 'data\S001\S001R06.edf'

# print(process_subject(subjects[0]))

raw = mne.io.read_raw_edf(file_path, preload=True)

# Apply a bandpass filter (1-40 Hz) to the raw data
raw = raw.copy().notch_filter(freqs=60.0)
raw_filtered = raw.copy().filter(2, None, method='iir')

# Re-reference the data to the average of all channels
raw_filtered.set_eeg_reference('average', projection=True)

events, event_id = mne.events_from_annotations(raw)
tmin, tmax = 0, 4.1  # Adjust these values based on your experiment protocol

# Create epochs based on events
epochs = mne.Epochs(raw_filtered, events, event_id, tmin, tmax, proj=False, baseline=(0,4.1), preload=True)

events, event_id = mne.events_from_annotations(raw_filtered)

# Print basic information about the file
info = raw_filtered.info

annotations = raw_filtered.annotations

print(annotations.duration)

# Extract annotations details
annotations_df = annotations.to_data_frame()

# print(annotations_df)


# Plot the raw EEG data
raw_filtered.plot(n_channels=10, duration=10, scalings='auto')
plt.show()