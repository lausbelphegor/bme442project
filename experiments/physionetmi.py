import moabb
import torch
from braindecode import EEGClassifier
from braindecode.models import EEGNetv4, EEGConformer, ATCNet, EEGITNet, EEGInception
from braindecode.util import set_random_seeds
from skorch.callbacks import LRScheduler, Checkpoint, TrainEndCheckpoint
from skorch.helper import predefined_split
from torch.utils.data import ConcatDataset
from torchinfo import summary
from skorch import NeuralNetClassifier
from skorch.helper import SliceDataset

import dataset_loader
import models
import utils
from models import cuda

moabb.set_log_level("info")

class PhysionetMIExperiment:
    def __init__(self, args, config, logger):
        # Clip to create window dataset
        if config['dataset']['n_classes'] == 3:
            events_mapping = {'left_hand': 0, 'right_hand': 1, 'feet': 2}
        else:
            events_mapping = {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'hands': 3}
        self.windows_dataset = self.ds.create_windows_dataset(
            trial_start_offset_seconds=-1,
            trial_stop_offset_seconds=1,
            mapping=events_mapping
        )

        self.n_channels = self.ds.get_channel_num()
        self.n_times = self.ds.get_input_window_sample()
        self.n_classes = config['dataset']['n_classes']
        # Training routine
        self.n_epochs = config['fit']['epochs']
        self.lr = config['fit']['lr']
        self.batch_size = config['fit']['batch_size']

        # User options
        self.save = args.save
        self.save_dir = args.save_dir
        self.strategy = args.strategy
        self.model_name = args.model
        self.verbose = config['fit']['verbose']

        self.logger = logger

    def __get_classifier(self):
        callbacks = [("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=self.n_epochs - 1))]
        if self.model_name == 'EEGNet':
            return NeuralNetClassifier(module=EEGNetv4,
                                       module__in_chans=self.n_channels,
                                       module__n_classes=self.n_classes,
                                       module__input_window_samples=self.n_times,
                                       module__kernel_length=32,
                                       module__drop_prob=0.5,
                                       criterion=torch.nn.CrossEntropyLoss,
                                       optimizer=torch.optim.Adam,
                                       optimizer__lr=self.lr,
                                       train_split=None,
                                       iterator_train__shuffle=True,
                                       batch_size=self.batch_size,
                                       callbacks=callbacks,
                                       device='cuda' if cuda else 'cpu',
                                       verbose=self.verbose
                                       )
        elif self.model_name == 'EEGConformer':
            return NeuralNetClassifier(module=EEGConformer,
                                       module__n_chans=self.n_channels,
                                       module__n_outputs=self.n_classes,
                                       module__n_times=self.n_times,
                                       module__final_fc_length='auto',
                                       module__add_log_softmax=False,
                                       criterion=torch.nn.CrossEntropyLoss,
                                       optimizer=torch.optim.Adam,
                                       optimizer__betas=(0.5, 0.999),
                                       optimizer__lr=self.lr,
                                       train_split=None,
                                       iterator_train__shuffle=True,
                                       batch_size=self.batch_size,
                                       callbacks=callbacks,
                                       device='cuda' if cuda else 'cpu',
                                       verbose=self.verbose
                                       )
        else:
            raise ValueError(f"model {self.model_name} is not supported on this dataset.")

    def __get_subject_split(self):
        all_valid_subjects = []
        train_subjects = []
        test_subjects = []
        for i in range(1, 110):
            if i not in [88, 89, 92, 100]:
                all_valid_subjects.append(i)
                if i <= 84:
                    train_subjects.append(i)
                else:
                    test_subjects.append(i)
        return all_valid_subjects, train_subjects, test_subjects

    def __get_subjects_datasets(self, dataset_split_by_subject, split_subject):
        valid_dataset = []
        for i in split_subject:
            for ds in dataset_split_by_subject[str(i)].datasets:
                if 'left_hand' in ds.windows.event_id or 'right_hand' in ds.windows.event_id:
                    valid_dataset.append(ds)
        split_datasets = ConcatDataset(valid_dataset)
        return split_datasets

    def __within_subject_experiment(self):
        # Split dataset for single subject
        subjects_windows_dataset = self.windows_dataset.split('subject')
        n_subjects = len(subjects_windows_dataset.items())
        avg_accuracy = 0
        for subject, windows_dataset in subjects_windows_dataset.items():
            # Evaluate the model by test accuracy for "Hold-Out" strategy
            train_dataset = windows_dataset.split('session')['0train']
            test_dataset = windows_dataset.split('session')['1test']
            train_X = SliceDataset(train_dataset, idx=0)
            train_y = SliceDataset(train_dataset, idx=1)
            test_X = SliceDataset(test_dataset, idx=0)
            test_y = SliceDataset(test_dataset, idx=1)
            clf = self.__get_classifier()
            # Save the last epoch model for test
            if self.save:
                clf.callbacks.append(TrainEndCheckpoint(dirname=self.save_dir + f'\\S{subject}'))
            clf.fit(train_X, y=train_y, epochs=self.n_epochs)
            # Calculate test accuracy for subject
            test_accuracy = clf.score(test_X, y=test_y)
            avg_accuracy += test_accuracy
            self.logger.info(f"Subject{subject} test accuracy: {(test_accuracy * 100):.4f}%")
        self.logger.info(f"Average test accuracy: {(avg_accuracy / n_subjects * 100):.4f}%")

    def __cross_subject_experiment(self):
        set_random_seeds(seed=self.config['fit']['seed'], cuda=cuda)
        _, train_subjects, test_subjects = self.__get_subject_split()
        dataset_split_by_subject = self.windows_dataset.split('subject')
        train_set = self.__get_subjects_datasets(dataset_split_by_subject, train_subjects)
        test_set = self.__get_subjects_datasets(dataset_split_by_subject, test_subjects)
        clf = self.__get_classifier()
        clf.train_split = predefined_split(test_set)
        clf.fit(X=train_set, y=None, epochs=self.n_epochs)

    def run(self):
        if self.strategy == 'within-subject':
            self.__within_subject_experiment()
        elif self.strategy == 'cross-subject':
            self.__cross_subject_experiment()

# def __get_subject_split():
#     all_valid_subjects = []
#     train_subjects = []
#     test_subjects = []
#     for i in range(1, 110):
#         if i not in [88, 89, 92, 100]:
#             all_valid_subjects.append(i)
#             if i <= 84:
#                 train_subjects.append(i)
#             else:
#                 test_subjects.append(i)
#     return all_valid_subjects, train_subjects, test_subjects


# def __get_subjects_datasets(dataset_split_by_subject, split_subject, n_classes):
#     if n_classes == 2:
#         valid_dataset = []
#         for i in split_subject:
#             for ds in dataset_split_by_subject[str(i)].datasets:
#                 if 'left_hand' in ds.windows.event_id or 'right_hand' in ds.windows.event_id:
#                     valid_dataset.append(ds)
#         split_datasets = ConcatDataset(valid_dataset)
#     else:
#         split_datasets = ConcatDataset([dataset_split_by_subject[str(i)] for i in split_subject])
#     return split_datasets


# def physionet(args, config):
#     set_random_seeds(seed=config['fit']['seed'], cuda=cuda)
#     all_valid_subjects, _, _ = __get_subject_split()
#     ds = dataset_loader.DatasetFromBraindecode('physionet', subject_ids=all_valid_subjects)
#     ds.uniform_duration(4.0)
#     ds.drop_last_annotation()
#     ds.preprocess(resample_freq=config['dataset']['resample'], high_freq=config['dataset']['high_freq'],
#                   low_freq=config['dataset']['low_freq'], picked_channels=config['dataset']['channels'])
#     channels_name = ds.get_channels_name()
#     print(channels_name)
#     n_classes = config['dataset']['n_classes']
#     if n_classes == 3:
#         events_mapping = {
#             'left_hand': 0,
#             'right_hand': 1,
#             'feet': 2
#         }
#     else:
#         events_mapping = {
#             'left_hand': 0,
#             'right_hand': 1,
#             'feet': 2,
#             'hands': 3
#         }
#     windows_dataset = ds.create_windows_dataset(trial_start_offset_seconds=-1, trial_stop_offset_seconds=1,
#                                                 mapping=events_mapping)
#     n_channels = ds.get_channel_num()
#     input_window_samples = ds.get_input_window_sample()
#     if args.model == 'EEGNet':
#         model = EEGNetv4(in_chans=n_channels, n_classes=n_classes,
#                          input_window_samples=input_window_samples, kernel_length=32, drop_prob=0.5)
#     elif args.model == 'ASTGCN':
#         model = nn_models.ASTGCN(n_channels=n_channels, n_classes=4, input_window_size=input_window_samples,
#                                  kernel_length=32)
#     elif args.model == 'BaseCNN':
#         model = nn_models.BaseCNN(n_channels=n_channels, n_classes=n_classes, input_window_size=input_window_samples)
#     else:
#         raise ValueError(f"model {args.model} is not supported on this dataset.")

#     if cuda:
#         model.cuda()
#     summary(model, (1, n_channels, input_window_samples, 1))

#     n_epochs = config['fit']['epochs']
#     lr = config['fit']['lr']
#     batch_size = config['fit']['batch_size']
#     callbacks = [("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1))]
#     if args.save:
#         callbacks.append(Checkpoint(monitor='valid_acc_best', dirname=args.save_dir,
#                                     f_params='{last_epoch[valid_accuracy]}.pt'))
#     if args.selection:
#         callbacks.append(("get_electrode_importance", utils.GetElectrodeImportance()))
#     clf = EEGClassifier(module=model,
#                         iterator_train__shuffle=True,
#                         criterion=torch.nn.CrossEntropyLoss,
#                         optimizer=torch.optim.Adam,
#                         train_split=None,
#                         optimizer__lr=lr,
#                         batch_size=batch_size,
#                         callbacks=callbacks,
#                         device='cuda' if cuda else 'cpu'
#                         )
#     dataset_split_by_subject = windows_dataset.split('subject')
#     _, train_subjects, test_subjects = __get_subject_split()
#     train_set = __get_subjects_datasets(dataset_split_by_subject, train_subjects, n_classes)
#     test_set = __get_subjects_datasets(dataset_split_by_subject, test_subjects, n_classes)
#     clf.train_split = predefined_split(test_set)
#     clf.fit(X=train_set, y=None, epochs=n_epochs)
