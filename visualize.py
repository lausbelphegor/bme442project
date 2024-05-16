import matplotlib.pyplot as plt
import mne

# Load the EDF file
file_path = 'data\S001\S001R02.edf'
raw = mne.io.read_raw_edf(file_path, preload=True)

# Apply a bandpass filter (1-40 Hz) to the raw data
raw = raw.copy().notch_filter(freqs=60.0)
raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=40.0)

# Re-reference the data to the average of all channels
raw_filtered.set_eeg_reference('average', projection=True)

# Print basic information about the file
info = raw_filtered.info

annotations = raw_filtered.annotations


# Extract annotations details
annotations_df = annotations.to_data_frame()

print(annotations_df)


# Plot the raw EEG data
raw_filtered.plot(n_channels=10, duration=10, scalings='auto')
plt.show()