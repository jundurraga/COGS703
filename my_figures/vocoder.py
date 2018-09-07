from scipy.io.wavfile import read
import os
from HRTF.processing_tools import *



# a_file = '/home/jundurraga/pCloudDrive/Documents/Presentations/MAPS2018/audio/sentences/arctic_a0002.wav'
a_file = "/home/jundurraga/pCloudDrive/Documents/Presentations/MAPS2018/audio/cheap.wav"
fig_path = '/home/jundurraga/pCloudDrive/Documents/Presentations/MAPS2018/my_figures/'

n_ch = 6
fc, frange = spacing_log(120., 3000., n_ch)
[fs, data] = read(a_file)
if np.ndim(data) > 1:
    data = data[:, 0]
data = data.astype(np.float)
data /= np.max(data)
fs = float(fs)

### Parameters ###
fft_size = 2048 # window size for the FFT
step_size = int(fft_size/16) # distance to slide along the window (in time)
spec_thresh = 4.0  # threshold for spectrograms (lower filters out more noise)

data -= np.mean(data)
freqs, time, stfft_mag = spectrogram(data,
                                     window=windows.hamming(fft_size),
                                     nperseg=fft_size,
                                     fs=fs,
                                     nfft=fft_size,
                                     noverlap=fft_size - step_size,
                                     mode='magnitude',
                                     scaling='density')
stfft_mag /= stfft_mag.max()
stfft_mag = np.log10(stfft_mag)
stfft_mag[stfft_mag < -spec_thresh] = -spec_thresh
inch = 2.54
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
row_idx = 0
gs = gridspec.GridSpec(2, 1)
ax1 = plt.subplot(gs[0, 0:])
t_data = np.arange(0, data.shape[0]) / fs
ax1.plot(t_data, data)
ax1.set_xlim(t_data[0], t_data[-1])
ax1.set_xticks([])
ax2 = plt.subplot(gs[1, 0:])
ax2.set_ylabel('Frequency [Hz]')
ax2.set_xlabel('Time [s]', fontsize=12)
ax2.set_xlim(t_data[0], t_data[-1])
cax = ax2.pcolormesh(time, freqs, stfft_mag, cmap=plt.cm.afmhot)
a_pos = ax2.get_position()
cbaxes = fig.add_axes([a_pos.x1 + 0.005, a_pos.y0, 0.005, a_pos.height])
cb = fig.colorbar(cax, cax=cbaxes)
# fig.colorbar(cax)
fig.suptitle('Original Spectrogram')
# ax2.set_ylim(80, 1000.)
ax2.set_title('', fontsize=12)
fig.subplots_adjust(hspace=0)
fig.savefig('{:s}spectrogram_{:s}.png'.format(
    fig_path,
    os.path.splitext(os.path.basename(a_file))[0]))
# plt.show()
