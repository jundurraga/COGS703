#!/usr/bin/env python
__author__ = "Jaime Undurraga"
__copyright__ = "Copyright 2018"
__credits__ = ["Jaime Undurraga"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Jaime Undurraga"
__email__ = "jaime.undurraga@gmail.com"
__status__ = "developing"
from HRTF.processing_tools import *
import matplotlib.pyplot as plt
import pysignal_generator.sin_functions as sg
from scipy.io.wavfile import read, write

#  Parameters
fs = 44100.
output_audio_path = '/home/jundurraga/pCloudDrive/Documents/Presentations/MAPS2018/audio/'
output_fig_path = '/home/jundurraga/pCloudDrive/Documents/Presentations/MAPS2018/my_figures/'
mf = 32.0
cf = 500.0
cp = 0

pul_1 = {'modulation_frequency': mf,
         'modulation_index': 1.0,
         'carrier_frequency': cf,
         'carrier_phase': 0,
         'modulation_phase': 0,
         'rise_fall_time': 0.01,
         'duration': 2.0}
pul_2 = {'modulation_frequency': mf,
         'modulation_index': 1.0,
         'carrier_frequency': cf,
         'carrier_phase': -np.pi / 2,
         'modulation_phase': np.pi / 2,
         'rise_fall_time': 0.01,
         'duration': 2.0}
y1, time = sg.sinusoidal(pul_1)
y2, time = sg.sinusoidal(pul_2)

y1 = np.concatenate((y1, np.flip(y1, axis=0)))
y2 = np.concatenate((y2, np.flip(y2, axis=0)))

# save plot
fig_file = '{:s}_d_type_{:s}_p1_{:.2f}_p2_{:.2f}_mi_{:.1f}_mp1_{:.2f}_mp2_{:.2f}_cf_{:.1f}_mf_{:.1f}.png'.format(
    output_fig_path,
    'low_500',
    pul_1['carrier_phase'],
    pul_2['carrier_phase'],
    pul_2['modulation_index'],
    pul_1['modulation_phase'],
    pul_2['modulation_phase'],
    pul_2['carrier_frequency'],
    pul_2['modulation_frequency']
)
fig = plt.figure(fig_file)
ax = fig.add_subplot(111)
time = np.arange(0, y1.shape[0]) / fs
ax.plot(time, y1)
ax.plot(time, y2)
ax.set_xlim(1.96, 2.04)
ax.set_xlabel('Time [s]')
ax.set_yticks([])

fig.savefig(fig_file)


mixed_data = np.hstack((y1, y2))
scaled = np.int16(mixed_data/np.max(np.abs(mixed_data), axis=0) * 32767)

output_audio_file = '{:s}_d_type_{:s}_p1_{:.2f}_p2_{:.2f}_mi_{:.1f}_mp1_{:.2f}_mp2_{:.2f}_cf_{:.1f}_mf_{:.1f}.wav'.format(
    output_audio_path,
    'low_500',
    pul_1['carrier_phase'],
    pul_2['carrier_phase'],
    pul_2['modulation_index'],
    pul_1['modulation_phase'],
    pul_2['modulation_phase'],
    pul_2['carrier_frequency'],
    pul_2['modulation_frequency']
)
#
write(output_audio_file, int(fs), scaled)

