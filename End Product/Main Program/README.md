## MAIN PROGRAM RUNNING IN PYTHON 3 

## Things to prepare
<br>
These files will run on Pi Zero / Pi 3B+ / etc through Command Line
python 3 /This/CodePath/runThisCode.py
<br>
on Boot an excecution file will tell pi to run this code (.sh)

<br>
recording audio through USB Sound Card requires some settings tweaking on the default sound card


Must make sure the requirements are already installed as listed below


## Requirements
Make Sure everything is installed and ready
<br>
from keras.models import Sequential<br>
from keras.layers import Dense<br>
from keras.models import model_from_json<br>
import os<br>
import wave, sys, pyaudio<br>
import contextlib<br>
from pydub import AudioSegment<br>
import scipy.io.wavfile as wav<br>
import speechpy<br>
import wave<br>
import numpy as np<br>
from sklearn.metrics import classification_report<br>


