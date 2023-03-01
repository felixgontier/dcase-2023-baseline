import glob
import os
import pickle

import h5py
import librosa
import numpy as np


def log_mel_spectrogram(y,
                        sample_rate=44100,
                        window_length_secs=0.025,
                        hop_length_secs=0.010,
                        num_mels=128,
                        log_offset=0.0):
    """Convert waveform to a log magnitude mel-frequency spectrogram.

    :param y: 1D np.array of waveform data.
    :param sample_rate: The sampling rate of data.
    :param window_length_secs: Duration of each window to analyze.
    :param hop_length_secs: Advance between successive analysis windows.
    :param num_mels: Number of Mel bands.
    :param fmin: Lower bound on the frequencies to be included in the mel spectrum.
    :param fmax: The desired top edge of the highest frequency band.
    :param log_offset: Add this to values when taking log to avoid -Infs.
    :return:
    """
    window_length = int(round(sample_rate * window_length_secs))
    hop_length = int(round(sample_rate * hop_length_secs))
    fft_length = 2 ** int(np.ceil(np.log(window_length) / np.log(2.0)))

    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_fft=fft_length, hop_length=hop_length,
                                                     win_length=window_length, n_mels=num_mels)

    return np.log(mel_spectrogram + log_offset)


# %%

global_params = {
    "dataset_dir": "data",
    "audio_splits": ["development", "validation", "evaluation", "test"]
}

# Load audio info
audio_info = os.path.join(global_params["dataset_dir"], "audio_info.pkl")
with open(audio_info, "rb") as store:
    audio_fid2fname = pickle.load(store)["audio_fid2fname"]

# Extract log mel for splits
for split in global_params["audio_splits"]:

    fid2fname = audio_fid2fname[split]
    fname2fid = {fid2fname[fid]: fid for fid in fid2fname}

    audio_dir = os.path.join(global_params["dataset_dir"], split)
    audio_logmel = os.path.join(global_params["dataset_dir"], f"{split}_audio_logmels.hdf5")

    with h5py.File(audio_logmel, "w") as stream:

        for fpath in glob.glob(r"{}/*.wav".format(audio_dir)):
            try:
                fname = os.path.basename(fpath)
                fid = fname2fid[fname]

                y, sr = librosa.load(fpath, sr=None, mono=True)
                log_mel = log_mel_spectrogram(y=y, sample_rate=sr, window_length_secs=0.040, hop_length_secs=0.020,
                                              num_mels=64, log_offset=np.spacing(1))

                stream[fid] = np.vstack(log_mel).transpose()  # [Time, Mel]
                print(fid, fname)
            except:
                print("Error audio file:", fpath)

    print("Save", audio_logmel)
