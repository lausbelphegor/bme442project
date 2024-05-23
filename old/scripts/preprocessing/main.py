import mne
import os
import numpy as np
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# https://github.com/OpenNeuroDatasets/ds003825
# (24648, 63, 276) = (epochs, channels, samples)

def preprocessing_things(partid):
    #ported
    # Specify the data path and create a directory for derivatives if it doesn't exist
    data_path = 'data'
    os.makedirs(os.path.join(data_path, 'derivatives', 'mne'), exist_ok=True)
    
    # Construct the file name for the continuous data
    cont_fn = os.path.join(data_path, 'derivatives', 'mne', f'sub-{partid:02d}_task-rsvp_continuous.fif')
    
    # Check if the continuous data file exists
    if os.path.isfile(cont_fn):
        logging.info(f'Using {cont_fn}')
        # Read the continuous data from the file
        raw = mne.io.read_raw_fif(cont_fn, preload=True)
    else:
        # Construct the file name for the raw data
        raw_fn = os.path.join(data_path, f'sub-{partid:02d}', 'eeg', f'sub-{partid:02d}_task-rsvp_eeg.vhdr')
        
        # Read the raw data from the file
        raw = mne.io.read_raw_brainvision(raw_fn, preload=True)
        
        # For participants 49 and 50, select the first 63 channels and set the reference to FCz if available
        if partid in [49, 50]:
            raw.pick_channels(raw.info['ch_names'][:63])
            if 'FCz' in raw.info['ch_names']:
                raw.set_eeg_reference(['FCz'], projection=False)
            else:
                logging.warning("FCz not found in the channels.")
        # For other participants, set the reference to Cz if available
        else:
            if 'Cz' in raw.info['ch_names']:
                raw.set_eeg_reference(['Cz'], projection=False)
            else:
                logging.warning("Cz not found in the channels.")
        
        # Apply a band-pass filter to the data
        raw.filter(0.1, 100)
        
        # Resample the data to 250 Hz
        raw.resample(250)
        
        # Save the preprocessed data to a file
        raw.save(cont_fn, overwrite=True)
    
    # Load the events data from a CSV file
    events_csv = os.path.join(data_path, f'sub-{partid:02d}', 'eeg', f'sub-{partid:02d}_task-rsvp_events.csv')
    events_df = pd.read_csv(events_csv)
    
    # Convert the stimulus onset times to samples
    stim_on = (events_df['time_stimon'].values * raw.info['sfreq']).astype(int)
    
    # Create an events array with stimulus onset times and trigger values
    events = np.column_stack((stim_on, np.zeros_like(stim_on, dtype=int), np.ones_like(stim_on, dtype=int)))
    
    # Create an epochs object from the raw data and events
    epochs = mne.Epochs(raw, events, event_id={'stimulus': 1}, tmin=-0.1, tmax=1.0, preload=True)
    
    # Save the epochs to a file
    epochs.save(os.path.join(data_path, 'derivatives', 'mne', f'sub-{partid:02d}_task-rsvp-epo.fif'), overwrite=False)
    
    logging.info(f'Finished sub-{partid:02d}.')

# Call the preprocessing function for participant 2
# preprocessing_things(2)

# Load the epochs data from a file
epochs = mne.read_epochs('data/derivatives/mne/sub-02_task-rsvp-epo.fif')

# Get the epochs data as a 3D array
epochs_data = epochs.get_data()
logging.info(f'Epochs data shape: {epochs_data.shape}')

# Plot a sample of the raw data
# epochs.plot(n_epochs=1, n_channels=64, scalings='auto', title='Sample Raw Data')

# Plot the average of the epochs data
# epochs.plot_image(combine='mean', vmin=-200, vmax=200, cmap='viridis')

# Plot the power spectral density of the epochs data
# fig_psd = epochs.plot_psd(fmin=0.5, fmax=50, average=True, spatial_colors=True)

# Plot the topomap of the average of the epochs data
# fig_topomap = epochs.average().plot_topomap(times=[0.1], size=3, title='Topomap at 100 ms',