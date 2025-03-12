import os
import sys
from fnmatch import fnmatch

import audiofile
import opensmile
import pandas as pd
from tqdm import tqdm

# Feature extraction methods


def opensmile_feature_extraction(
    path_audio,
    path_audio_pattern="*.wav",
    path_output=None,
    smile_feature_set=opensmile.FeatureSet.eGeMAPSv02,
    smile_feature_level=opensmile.FeatureLevel.Functionals,
    read_always_2d=True,
    window_size=0.25,
    step=0.25,
) -> None:
    """
    Extracting eGeMAPS features from audio files. Saving features as CSV files

    Args:
        path_audio (str): path to audio/feature/input files
        path_audio_pattern (str): audio/feature/input file pattern to import
        path_output (str): optional, the output directory. By default, it will save into the path_audio
        smile_feature_set (enum): opensmile feature set to use (see opensmile.FeatureSet)
        smile_feature_level (enum): opensmile feature level to use (see opensmile.FeatureLevel)
        read_always_2d (bool): corresponding to the always_2d argument given to the audiofile.read method (see audiofile.read)
        window_size (float): window_size in second
        step (float): step in second

    """
    # Getting all audio files
    audio = []
    for path, _, files in os.walk(path_audio):
        for name in files:
            if not name.startswith(".") and fnmatch(name, path_audio_pattern):
                audio.append(os.path.join(path, name))

    # Setting OpenSMILE
    smile = opensmile.Smile(
        feature_set=smile_feature_set,
        feature_level=smile_feature_level,
    )

    # Processing each file
    for file in tqdm(audio, colour="#5dbfc9"):
        features = []
        # Reading audio file
        signal, sampling_rate = audiofile.read(file, always_2d=read_always_2d)
        # Applying a sliding window to the audio
        max_signal_len = len(signal[0]) / sampling_rate
        for step_current in tqdm(
            range(int(max_signal_len / step)),
            colour="#5dbfc9",
            desc=os.path.basename(file),
        ):
            features.append(
                smile.process_signal(
                    signal,
                    sampling_rate,
                    start=step * step_current,
                    end=step * step_current + window_size,
                )
            )
        # Merging in one DataFrame
        out = pd.concat(features)
        # Outputting as a csv file
        out.to_csv(
            os.path.join(
                path_audio if not path_output else path_output,
                os.path.splitext(os.path.basename(file))[0] + ".csv",
            ),
            index=False,
        )


if __name__ == "__main__":
    opensmile_feature_extraction(os.path.realpath(sys.argv[1]))
