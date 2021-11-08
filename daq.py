"""Capture microphone signals using Chung's 4ch NI USB daq

NI 9234 in NI USB 9162 carrier.

Based on sound.py

The highest flex modes we are likely to want to capture are <10 kHz so
need a sample rate of >20 kHz. The USB-9234 supports 25.6 kHz, so lets
use that.


Refs:
https://www.ni.com/documentation/en/ni-daqmx/20.1/mxcncpts/datatrans/


jh, Nov 2021
"""

import nidaqmx
import nidaqmx.stream_readers
import nidaqmx.constants

#from nidaqmx._task_modules.read_functions import _read_analog_f_64

from threading import Thread

import wx
from pubsub import pub

import logging
logger = logging.getLogger(__name__)

import os
import sys
import time

import numpy as np
from matplotlib import pyplot as plt

mic_lines = (
    'Dev1/ai0',
    'Dev1/ai1',
    'Dev1/ai2',
    'Dev1/ai3',
)
BUF = None

class DAQThread(Thread):

    def __init__(self, sample_rate=25600, rolling_period=0.1, trigger_level=10):
        Thread.__init__(self)
        self.running = False
        bufsize = int(rolling_period * sample_rate)
        self.task = nidaqmx.Task('rolling_acquire')
        for mic_line in mic_lines:
            self.task.ai_channels.add_ai_microphone_chan(
                mic_line,
                units = nidaqmx.constants.SoundPressureUnits.PA,
                mic_sensitivity=10.0,
                max_snd_press_level=100.0,
                current_excit_val=2.e-3,
            )
        self.task.timing.cfg_samp_clk_timing(
            sample_rate,
            sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
            samps_per_chan=bufsize*10
        )
        self.task.in_stream.input_buf_size = bufsize*10
        self.reader = nidaqmx.stream_readers.AnalogMultiChannelReader(self.task.in_stream)
        #self.buf = np.zeros((len(mic_lines), bufsize))
        global BUF
        BUF = np.zeros((len(mic_lines), bufsize))

        def read_callback(task_idx, event_type, num_samples, callback_data=None):
            #print(f'{num_samples}')
            #nret = self.reader.read_many_sample(self.buf, num_samples, timeout=0)
            nret = self.reader.read_many_sample(BUF, num_samples, timeout=0)
            #print(f'{nret}')
            if abs(BUF).max()>trigger_level:
                #self.running = False
                wx.CallAfter(pub.sendMessage, "DAQdata", data=nret)#self.buf)
                print(BUF.max())
            return 0

        self.task.register_every_n_samples_acquired_into_buffer_event(bufsize, read_callback)
        #time.sleep(0.1) # make sure the callback registered before continuing

    def run(self):
        #print('calling run()')
        self.task.start()
        #print('started')
        self.running = True
        while self.running:
            #print(f'running = {self.running} {self.task.in_stream.avail_samp_per_chan}')
            #time.sleep(1)
            pass
        self.task.stop()
        self.task.close()


def acquire(
        rolling_period=0.1,
        sample_rate=25600,
        trigger_level=None,
        pretrigger_time=1e-3,
        posttrigger_time=100e-3,
        display=True):
    """Acquire microphone data with software triggering
    """
    import nidaqmx
    import time

    data = None
    pretrigger_samples = int(np.ceil(pretrigger_time * sample_rate))
    posttrigger_samples = int(np.ceil(posttrigger_time * sample_rate))
    acq_samples = pretrigger_samples + posttrigger_samples
    bufsize = int(np.ceil(rolling_period * sample_rate))

    sys.stdout.write(f'setting trigger @ {trigger_level}\n')
    sys.stdout.write(f' ({pretrigger_samples} pre, {posttrigger_samples} post)\n')

    with nidaqmx.Task('rolling_acquire') as task:
        for mic_line in mic_lines:
            task.ai_channels.add_ai_microphone_chan(
                mic_line,
                units = nidaqmx.constants.SoundPressureUnits.PA,
                #terminal_config=<TerminalConfiguration.DEFAULT: -1>,
                mic_sensitivity=10.0,
                max_snd_press_level=100.0,
                #current_excit_source=<ExcitationSource.INTERNAL: 10200>,
                current_excit_val=2.e-3,
                #custom_scale_name=u''
            )
        task.timing.cfg_samp_clk_timing(
            sample_rate,
            sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
            samps_per_chan=bufsize*100)
        #task.in_stream.input_buf_size = bufsize*100

        reader = nidaqmx.stream_readers.AnalogMultiChannelReader(task.in_stream)
        buf = np.zeros((len(mic_lines), bufsize))
        running = False
        acquiring = False

        def read_callback(task_idx, event_type, num_samples, callback_data=None):
            nonlocal running
            nonlocal buf
            nonlocal data
            nonlocal acquiring

            nret = reader.read_many_sample(buf, num_samples, timeout=0)
            sys.stdout.write('.')
            sys.stdout.flush()

            abs_max = abs(buf).max()
            # if no part of the signal is above the trigger level then keep only the
            # tail of the signal in case the next buffer rises immediately
            if not acquiring and abs_max<trigger_level:
                data = buf[:,-pretrigger_samples:].copy()
            else:
                # right so we've been triggered...
                acquiring = True

                # append the buffer to data
                data = np.hstack((data, buf))

                # if we've collected all the samples we need we can stop
                if data.shape[1]>acq_samples:
                    running = False

            return 0

        task.register_every_n_samples_acquired_into_buffer_event(bufsize, read_callback)
        #time.sleep(0.1)
        task.start()
        running = True
        while running:
            pass
        time.sleep(0.1) # pause for a bit to let any pending captures complete

    _, j = np.where(abs(data)>trigger_level)
    data = data[:,max(j[0]-pretrigger_samples, 0):min(data.shape[1], j[0]+posttrigger_samples)]
    sys.stdout.write(f'{abs(data).max():0.1f}')
    t = (np.arange(data.shape[1])-pretrigger_samples)/sample_rate

    if display:
        fig = plt.figure()
        ax_sig = fig.add_subplot(211)
        ax_fft = fig.add_subplot(212)
        ax_sig.cla()
        ax_sig.plot(t, data.T)
        ax_sig.set_xlim((-0.003,0.03))
        ax_sig.grid()
        ax_fft.cla()
        for chan in range(data.shape[0]):
            from scipy.signal import periodogram
            f, spec = periodogram(data[chan,:], nfft=2**22, fs=sample_rate, scaling='spectrum')
            ax_fft.semilogy(f, np.sqrt(spec))
        ax_fft.set_xlim((0,500))
        ax_fft.set_ylim((0.01,10))
        ax_fft.grid()
        plt.show()

    return t, data


if __name__=="__main__":
    # threaded version for use with GUI - slow and maybe ***BROKEN***
    #T = DAQThread().start()

    # cmdline version
    # daq key1=val key2=val ... ID
    import datastore
    import json
    assert len(sys.argv)>1, "Missing ID"
    signalid = sys.argv[-1]
    sys.stdout.write(f"{signalid}")
    metafn = 'meta.json'
    meta = {
        'h5fn': 'signals.h5',
        'rolling_period': 0.1,
        'sample_rate': 25600,
        'trigger_level': 10,
    }
    if os.path.isfile(metafn):
        with open(metafn, 'r') as f:
            meta = meta | json.load(f)
    for arg in sys.argv[1:-1]:
        try:
            k, v = arg.split("=")
        except:
            sys.stderr.write(f"*** Failed to parse {arg}")
        meta[k] = v
    #print(meta)
    t, data = acquire(
        rolling_period=meta['rolling_period'],
        sample_rate=meta['sample_rate'],
        trigger_level=meta['trigger_level'],
        display=True
    )
    for chan, gain in enumerate(meta['gain']):
        data[chan, :] *= gain
    uuid = datastore.write(
        meta['h5fn'],
        signalid,
        signal=np.vstack((data, t)).T,
        meta=meta
    )
    sys.stdout.write(f' => {uuid}\n')
