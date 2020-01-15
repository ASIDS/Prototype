## MAIN PROGRAM RUNNING IN PYTHON 3 

The program consist of simple instructions of in a loop 
<br>1.Record Audio
<br>Using USB Sound Card (Raspberry pi doesnt have it built-in)


<br>2.Preprocess Slice Audio, Convert to Single Channel


<br>3.Extract Features of each slice

<br>4.Load Model -> Produce Inference
<br>Load model 2 files json and h5 can be replaced with new version of model trained later just only if the INPUT SIZE 98,40 AND OUTPUT CLASS = 4, ARE EXACTLY THE SAME 
<br>
<br>5.Compile Inference -> Create a Final Decision Filter
<br Inferences are also restricted to the Classes = 4 (MUST MODIFY FOR DIFFERENT MODEL/CLASS NUMBERS)

<br>6.Sound Alarm -> LoRa Message -> Gateway
<br>7.Sleep Until Next Routine, Redo Steps 1-7 each time device wake up


## Things to prepare
<br>
These files will run on Pi Zero / Pi 3B+ / etc through Command Line
<br>
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


