import numpy as np
import cochlea
from scipy.io.wavfile import read
import os
import thorns.waves as wv
import auditory_models.tools.analysis_tools as at
import samplerate
from HRTF.processing_tools import *


def audio_to_gbc(fs=np.float,
                 input_sound=np.array([]),
                 anf_num=(4, 3, 2),
                 l_freq=None,
                 h_freq=None):
    cochlea_length = 35.0
    basilar_membrane_pos_high = np.log10(h_freq / 165.4 + 0.88) * cochlea_length / 2.1
    basilar_membrane_pos_low = np.log10(l_freq / 165.4 + 0.88) * cochlea_length / 2.1
    total_ihc = 500.0
    n_ihc = int(max(1, round(total_ihc * (basilar_membrane_pos_high - basilar_membrane_pos_low) / cochlea_length)))

    anf_spikes = cochlea.run_zilany2014(
        input_sound,
        fs,
        anf_num=anf_num,
        cf=(l_freq, h_freq, round(n_ihc)),
        species='human',
        seed=None,
    )
    anf_spikes = anf_spikes.sort_values(by='cf')
    return anf_spikes


fs = 100000
l_freq = 130.0
h_freq = 16000.
spl = 65.0
a_file = "/home/jundurraga/pCloudDrive/Documents/Presentations/MAPS2018/audio/cheap_clean.wav"
root_path = "/home/jundurraga/pCloudDrive/Documents/Presentations/MAPS2018/"
[fsf, data] = read(a_file)
if np.ndim(data) > 1:
    data = data[:, 0]

ratio = fs / fsf
converter = 'sinc_best'
data = samplerate.resample(data, ratio, converter)
data = data.astype(np.float)
sound = wv.set_dbspl(np.squeeze(data), spl)
anf = audio_to_gbc(
    fs=fs,
    l_freq=l_freq,
    h_freq=h_freq,
    input_sound=sound
)

figures_path = root_path + 'figures_an'
fig = at.plot_raster(data=anf,
                     plot_function=at.plot_raster,
                     figure_name_by=['type'],
                     group_factors=['type'],
                     split_by='duration'
                     )
ax = fig.get_axes()[0]
ax.set_title('')
fig.suptitle('')
fig.set_size_inches(12, 9)
fig.savefig(root_path + 'my_figures/rater.png')
#  Parameters
fft_size = 2048  # window size for the FFT
step_size = int(fft_size / 16)  # distance to slide along the window (in time)
spec_thresh = 4.0
fig1, s1, time, freqs = plot_spectrogram(data,
                                         fs=fs,
                                         fft_size=fft_size,
                                         step_size=step_size,
                                         spec_thresh=spec_thresh,
                                         ylim=(0, 16000.))
fig1.set_size_inches(14.5, 9)
fig1.savefig(root_path + 'my_figures/spectrogram_cheap.png')
