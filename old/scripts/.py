import mne
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

path = 'C:\\temp\\BME_442\\bme442project\\data\\sub-02\\eeg'
raw = mne.io.read_raw_brainvision(os.path.join(path,'sub-02_task-rsvp_eeg.vhdr'), preload=True)

# Load the events.tsv file
events = pd.read_csv(os.path.join(path,'sub-02_task-rsvp_events.tsv'), delimiter='\t')


raw.plot()

# Load and process events
events_df = pd.read_csv(os.path.join(path,'sub-02_task-rsvp_events.csv'))
events_df['sample_index'] = (events_df['time_stimon'] * raw.info['sfreq']).astype(int)

# Create MNE-compatible events array
mne_events = np.column_stack((events_df['sample_index'], np.zeros(len(events_df)), events_df['objectnumber']))

plt.show()