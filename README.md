README
=======

This repository contains code for preprocessing EEG data from the OpenNeuro dataset ds003825.

**Dependencies**

- Python 3.x
- MNE library (version 0.24.1 or later)
- NumPy library (version 1.20.0 or later)
- Pandas library (version 1.3.5 or later)
- Graphviz library (for neural network visualization)
- Logging library (for output management)

**Usage**

1. Clone the repository and navigate to the root directory.
2. Run the `preprocessing_things` function by calling `python preprocessing_things.py <partid>`, where `<partid>` is the participant ID (e.g., 2).
3. The script will preprocess the EEG data for the specified participant and save the results to a file.

**Files**

* `preprocessing_things.py`: Main script for preprocessing EEG data.
* `model.py`: Script for visualizing EEG Conformer network architecture.
* `data/`: Directory containing the EEG data files.
* `data/derivatives/mne/`: Directory containing the preprocessed EEG data files.
* `plots/`: Directory containing generated plots of the EEG data.

**Notes**

* The script assumes that the EEG data files are stored in the `data/` directory, with the filename format `sub-<partid>_task-rsvp_eeg.vhdr`.
* The script uses the MNE library to read and preprocess the EEG data.
* The preprocessed EEG data is saved to a file in the `data/derivatives/mne/` directory, with the filename format `sub-<partid>_task-rsvp-epo.fif`.
* The script also generates plots of the preprocessed EEG data, including a sample of the raw data, the average of the epochs data, and the power spectral density of the epochs data.

**Architecture Details:**

`Input Layer:` Batch of EEG trials.
`Temporal Convolution: Kernel size:` (1, 25), Activation: ELU.
`Spatial Convolution: Kernel size:` (ch, 1), Activation: ELU.
`Batch Normalization:` Standardizes input features by re-centering and scaling.
`Average Pooling:` Reduces dimensionality and computational load.
`Token Formation:` Prepares data for attention mechanism.
`Multi-Head Attention:` Enhances model's ability to focus on different parts of input data.
`Fully Connected Layers:` Maps the learned features to the output space.
`Output Layer:` Produces final classification.

**License**

This code is licensed under the MIT License. See the `LICENSE` file for details.