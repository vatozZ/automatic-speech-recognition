# automatic-speech-recognition
## Automatic Speech Recognition for some letters.
## This project is the ELEC367 Digital Signal Processing Final-Term Project, GTU

# Includes presentation in Turkish.->> sunum.ppsx

recording.py -----> Voice recording for the database is done here.

database (File) -----> The saved .wav files are stored here.

main.py ---------> reading, comparing and obtaining results are done here.

Audio_processing.py ----------> Processing sound, analyzing its properties in graphics (fft, stft, MFCCs, time space)
The program that needs to be run for.

Libraries to be imported:
import pyaudio
import wave
from python_speech_features import mfcc
import sys
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import math
from fastdtw import fastdtw (or) conda install -c bioconda / label / cf201901 fastdtw

You can import the necessary libraries using the pip install - command.
## Have fun 
