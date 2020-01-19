# Alternative Model - Random Forest Ensemble Using Scipy Library (BUT! Must train on End-product machine)
<br>
This Model requires less power to run and fir for Pi Zero that does not quite do well with tensorflow. But the downside is you must train the model on the specified machine to it can load with the same architecture.

<br>Model Must be Trained on Raspberry Pi(32 Bit System) in order to be able to load a model based on a 32 Bit Manageable
<br> ISSUE ! 64 Bit Model dont go well in 32 Bit System
<br>(https://stackoverflow.com/questions/47322937/unpacking-a-model-trained-with-64-bit-machine-on-a-32-bit-raspberry-pi)
<br>
Runtimes are very much faster and probably upgradable to real-Time monitoring
<br>
# Requirements
<br>import wave, sys, pyaudio
<br>import contextlib
<br>from pydub import AudioSegment
<br>import scipy.io.wavfile as wav
<br>import pickle; print("pickle", sys.version)
<br>import timeit; print("timeit", sys.version)
<br>import contextlib; print("contextlib", sys.version)
<br>import platform; print(platform.platform())
<br>import sys; print("Python", sys.version)
<br>import numpy as np; print("NumPy", np.__version__)
<br>import scipy; print("SciPy", scipy.__version__)
<br>import sklearn; print("Scikit-Learn", sklearn.__version__)
<br>import speechpy
<br>
OR
<br>
<br>pickle 3.7.3 (default, Apr 24 2019, 15:29:51) [MSC v.1915 64 bit (AMD64)]
<br>timeit 3.7.3 (default, Apr 24 2019, 15:29:51) [MSC v.1915 64 bit (AMD64)]
<br>contextlib 3.7.3 (default, Apr 24 2019, 15:29:51) [MSC v.1915 64 bit (AMD64)]
<br>Python 3.7.3 (default, Apr 24 2019, 15:29:51) [MSC v.1915 64 bit (AMD64)]
<br>NumPy 1.16.4
<br>SciPy 1.2.1
<br>Scikit-Learn 0.21.2
<br>speechpy 3.7.3 (default, Apr 24 2019, 15:29:51) [MSC v.1915 64 bit (AMD64)]

# Deployment Method
<br>
<br>1.Load Model (sav file) made using pickle library
<br>2.Record Audio (Same)
<br>3.Extract Feats (Same)
<br>4.Predict using predict or predict_proba scipy library


