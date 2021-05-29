import torch
import random
import numpy as np
from typing import Optional, Tuple, List

from audiomentations import (
    AddGaussianNoise,
    AddBackgroundNoise,
    ClippingDistortion,
    FrequencyMask,
    Gain,
    TimeStretch,
    PitchShift,
    Trim
)
from module.augmentations.reverberation import Reverberation


class SpeechAugment:
    def __init__(self,
        noise_dir=None,
        rir_dir=None,
        rir_lists=None,
        apply_prob=0.5,
        sample_rate=16000,
    ):
        self.sample_rate = sample_rate
        self.apply_prob = apply_prob
        self.transforms = [
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=1.0),
            ClippingDistortion(min_percentile_threshold=10, max_percentile_threshold=30, p=1.0),
            FrequencyMask(min_frequency_band=0.2, max_frequency_band=0.4, p=1.0),
            Gain(min_gain_in_db=-6, max_gain_in_db=6, p=1.0),
            TimeStretch(min_rate=0.9, max_rate=1.1, leave_length_unchanged=False, p=1.0),
            PitchShift(min_semitones=-2, max_semitones=3, p=1.0),
            Trim(p=1.0)
        ]
        if noise_dir is not None:
            self.transforms += [AddBackgroundNoise(sounds_path=noise_dir, min_snr_in_db=1, max_snr_in_db=5, p=1.0)]
        
        if rir_dir is not None and rir_lists is not None:
            self.transforms += [Reverberation(path_dir=rir_dir, rir_list_files=rir_lists, sampling_rate=sample_rate, p=1.0)]
        
        self.num_trans = len(self.transforms)

    def __call__(self, input_values: List[float]):
        """apply a random data augmentation technique from a list of transformations"""
        if random.random() < self.apply_prob:
            transform = self.transforms[random.randint(0, self.num_trans - 1)]
            input_values = transform(samples=np.array(input_values), sample_rate=self.sample_rate).tolist()
        return input_values
