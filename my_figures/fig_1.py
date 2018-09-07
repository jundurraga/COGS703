from matplotlib import pyplot as plt
import pysignal_generator.sin_functions as sg
import matplotlib
import numpy as np
import pyfftw
import multiprocessing
import matplotlib.animation as manimation
from scipy.io.wavfile import write
import soundfile as sf
import subprocess
import os

__author__ = 'jundurraga'

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='IPM-FR', artist='Matplotlib',
                comment='Movie support!')

file_path ='/home/jundurraga/Dropbox/Documents/Presentations/MAPS2018/my_figures/'
fs = 48000.0

def fftw_hilbert(x, axis=0):
    _fft = pyfftw.builders.fft(x, overwrite_input=False, planner_effort='FFTW_ESTIMATE', axis=axis,
                               threads=multiprocessing.cpu_count())
    fx = _fft()
    N = fx.shape[axis]
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    if len(x.shape) > 1:
        ind = [np.newaxis] * x.ndim
        ind[axis] = slice(None)
        h = h[ind]
    _ifft = pyfftw.builders.ifft(fx * h, overwrite_input=False, planner_effort='FFTW_ESTIMATE', axis=axis,
                                 threads=multiprocessing.cpu_count())
    return _ifft()

matplotlib.rcParams.update({'font.size': 8})

pl = {'ipd': -np.pi/8,
      'alt_ipd': np.pi/8,
      'amplitude':0.9,
      'modulation_index': 1.0,
      'modulation_frequency': 41.0,
      'carrier_frequency': 520.0,
      'cycle_rate': 16384.0,
      'fs': fs,
      'duration': 4.109375,
      'ipm_rate': 2.0,
      'include_triggers': 0,
      }

pulse, time = sg.sin_alt_ipd(pl)

fps_real = pl['modulation_frequency'] / 2.
fps = pl['modulation_frequency'] / 2.
writer = FFMpegWriter(fps=fps_real, metadata=metadata)


env = np.abs(fftw_hilbert(pulse, axis=0))
inch = 2.54
alt_period = pl['phase_alt_period'] / 2.0
n_cycles = time[-1] / alt_period
b = alt_period * pl['ipd']
ipd_factor_l = np.abs(pl['ipd']) / (np.pi / 2.0)
ipd_factor_r = np.abs(pl['alt_ipd']) / (np.pi / 2.0)
r = alt_period / 2.0 / 1.2

fig, ax2 = plt.subplots(1, 1, sharex=False)
fig.set_size_inches(10/inch, 10 / inch)
time_frame = np.linspace(0, pl['duration'], fps * pl['duration'])
window_duration = 0.05
video_file = file_path + "writer_test.mp4"
with writer.saving(fig, video_file, 300):
    for _i_frame in range(int(np.floor(pl['duration'] * fps))):
        _ini = np.argwhere(time >= _i_frame / fps)[0, 0]
        _end = np.argwhere(time < (_i_frame + 15) / fps)[-1, 0]
        _time = time[_ini:_end, :]
        idx_re = np.squeeze(np.floor(np.mod(_time / alt_period, 2)).astype(np.bool))
        idx_le = np.squeeze(np.logical_not(idx_re))
        t_el = _time[idx_le]
        t_er = _time[idx_re]
        _env = env[_ini:_end, :]
        _env_le = _env[idx_le, 0]
        _env_re = _env[idx_re, 1]

        # ax1.cla()
        # ax1.plot(_time, le[_ini:_end, :], 'b')
        # ax1.plot(_time, re[_ini:_end, :], 'r')
        # ax1.plot(t_el, _env_le, 'b')
        # ax1.plot(t_er, _env_re, 'r')
        # ax1.set_xlim(_time[0], _time[0] + window_duration)
        # # ax1.set_xlabel('Time [s]')
        # ax1.axvline(_time[0] + window_duration / 2.0)
        # plot nose
        ax2.cla()
        ax2.plot(_time[0] + b / 2.0, b / 2.0 + r, '^', color='gray', markersize=10, zorder=0)
        # plot head
        head = plt.Circle((_time[0] + b / 2.0, b / 2.0), r, color='gray')
        ax2.add_patch(head)
        # Plot sound source
        if _time[idx_le].any():
            cross_pos1 = np.argwhere(_time[idx_le] < _time[0] + window_duration / 2.0)
            cross_pos2 = np.argwhere(_time[idx_le] >= _time[0] + window_duration / 2.0)
            if cross_pos1.size and cross_pos2.size and (_time[idx_le][cross_pos2[0, 0]] -
                                                        _time[idx_le][cross_pos1[-1, 0]]) * fs < 2:
                ax2.plot(_time[0] + b / 2.0 + (r * ipd_factor_l) / 1.5, b / 2.0, 'o', color='r', markersize=8)
            else:
                ax2.plot(_time[0] + b / 2.0 - (r * ipd_factor_r) / 1.5, b / 2.0, 'o', color='b', markersize=8)
        else:
            ax2.plot(_time[0] + b / 2.0 - (r * ipd_factor_r) / 1.5, b / 2.0, 'o', color='b', markersize=8)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title('{:.1f} $^\circ$ / {:.1f} $^\circ$'.format(pl['ipd'] * 180/np.pi, pl['alt_ipd'] * 180/np.pi))
        fig.set_size_inches(3, 3)
        # plt.show()
        writer.grab_frame()

audio = pulse
# scaled = np.int16(audio/np.max(np.abs(audio), axis=0) * 32767)
audio_file = file_path + 'test.wav'
sf.write(audio_file, audio, int(fs))
# write(audio_file, int(fs), scaled)
data, samplerate = sf.read(audio_file)

video_file = file_path + "writer_test.mp4"
output_file = file_path + 'example{:.1f}_{:.1f}'.format(pl['ipd']*180/np.pi, pl['alt_ipd']*180/np.pi) + '.mp4'
if os.path.isfile(output_file):
    os.remove(output_file)
bashCommand = 'ffmpeg -i {:s} -i {:s} -c:v copy -ar {:d} -acodec mp3 {:s}'.format(video_file, audio_file, int(fs),output_file)
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
