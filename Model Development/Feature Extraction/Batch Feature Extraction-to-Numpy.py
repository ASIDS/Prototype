#!/usr/bin/env python
# coding: utf-8

# ## Initilize Feature Shape 
# ##### Empty Sets
# ##### DATASET = 500 Data of 40 x 173 features 
# ##### Target = 500

# In[25]:


import numpy as np
import librosa
totaldata =20000
##featureShape_x = z.shape[0]
##featureShape_y = z.shape[1]
featureShape_x = 98
featureShape_y = 40
dataset = np.ndarray(shape=(totaldata,featureShape_x,featureShape_y))
target =   np.ndarray(shape=(totaldata))

print ("Total data",totaldata)
print ("Features Shape X",featureShape_x)
print ("Features Shape Y",featureShape_y)
print ("Initial Dataset Empty Array Shape",dataset.shape)
print ("Initial Target Empty Array Shape",target.shape)


# # SELECT FILE TO EXTRACT ALL WAV FILE 
# 
# 

# In[3]:


import wave, sys, pyaudio
import contextlib
from pydub import AudioSegment
import os
import scipy.io.wavfile as wav
import speechpy
import contextlib
import wave
import numpy as np

import numpy 
from pydub import AudioSegment
import os
import librosa
path = 'C:/Users/Another/CNN/RAW/Validated/Logging Road'
path = 'C:/Users/Another/CNN/RAW/Validated/Combine Test'
path = 'C:/Users/Another/CNN/RAW/Validated/Combine Train'
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.WAV' in file:
            files.append(os.path.join(r, file))
            
            
totalDataWillBe = 0           
           
for f in files:
    if(f.__contains__("_BU")==0):
        if(1):
            #print(f)
            cutSeconds = 5
            slideSeconds = 0.25
            shift = int(cutSeconds/slideSeconds) - 1
            #print("Audio Shift seconds",slideSeconds)
            #print("Audio Shifts Segments",shift)

            if(shift == 0):
                shift = 1
            i =0
            para_fl = 0.1
            para_ovlp = 0.8


            totalFiles = 0
            for f in files:
                if(f.__contains__("_BU")==0):
                    if(1):
                        for z in range(shift): 
                            fname = f
                            with contextlib.closing(wave.open(fname,'r')) as faa:
                                frames = faa.getnframes()
                                rate = faa.getframerate()
                                duration = frames / float(rate)
                            segs =int(duration/cutSeconds)            
                            totalDataWillBe = totalDataWillBe + segs
                            
                            duration
                            
                            


# In[4]:


totalDataWillBe


# In[27]:


import wave, sys, pyaudio
import contextlib
from pydub import AudioSegment
import os
import scipy.io.wavfile as wav
import speechpy
import contextlib
import wave
import numpy as np
cutSeconds = 5
slideSeconds = 0.25
shift = int(cutSeconds/slideSeconds) - 1
print("Audio Shift seconds",slideSeconds)
print("Audio Shifts Segments",shift)

if(shift == 0):
    shift = 1
i =0
para_fl = 0.1
para_ovlp = 0.8


totalFiles = 0
for f in files:
    if(f.__contains__("_BU")==0):
        if(1):
            for z in range(shift): 
                fname = f
                with contextlib.closing(wave.open(fname,'r')) as faa:
                    frames = faa.getnframes()
                    rate = faa.getframerate()
                    duration = frames / float(rate)
                print("Reading File : ",fname,"With Target Label",1)
                print('SR',rate)
                print('duration',duration)
                segs =int(duration/cutSeconds)
                print('Segments',segs)    
                y = AudioSegment.from_wav(f)
                ShiftAudio = y[(slideSeconds*1000)*z:(cutSeconds*1000)*(segs)]
                segs = segs-1
                for sss in range(segs): 
                    newAudio = ShiftAudio[(cutSeconds*1000)*sss:(cutSeconds*1000)*(sss+1)]


                    newname = '/ZOOM'+f.split("\ZOOM")[1].split(".WAV")[0] + '_Shift' + str(z*int(slideSeconds*1000)) + 'ms_segments_'+ str(i) + '.WAV'
                    print('Extract Segment - ',newname)
                    ##COMMENT OUT SAVE FILE TO TEST RUN 
                    newAudio.export("TempDATA.WAV", format="wav")


                    fs, signal = wav.read("TempDATA.WAV")
                    print(f)
                    print(fs,"kHz")
                    print("Channels",signal.shape)
                    if(np.array_equal(signal.shape, [220500, ])):
                        print("1 Channel")
                    else:
                        signal = signal[0:44100*5,0]
                        print("2 Channels")

                    signal_preemphasized = speechpy.processing.preemphasis(signal, cof=0.98)
                    logenergy = speechpy.feature.lmfe(signal, sampling_frequency=fs, frame_length=para_fl, frame_stride=para_fl*0.5,
                    num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
                    
                    logenergy

                    dataset[i] = logenergy

                    if(f.__contains__("Ambience")):
                        target[i] = 0

                    elif(f.__contains__("Machete")):
                        target[i] = 1
                    elif(f.__contains__("Hatchet")):
                        target[i] = 1  
                    elif(f.__contains__("Chainsaw")):
                        target[i] = 2
                    else:
                        target[i] = 3

                    print("Data ",i)
                    print(f.split('CNN/')[1].split('\\')[0])
                    i += 1


# In[28]:


print (i,"Data Extracted")
meaningful = dataset[0:i][:][:]
meaningfultarget = target[0:i]
AAA = list(meaningfultarget).count(0)
HHH = list(meaningfultarget).count(1)

CCC = list(meaningfultarget).count(2)
VVV = list(meaningfultarget).count(3)
Aper = int(AAA/i*100)
Hper = int(HHH/i*100)
Cper = int(CCC/i*100)
Vper = int(VVV/i*100)
print(AAA,"samples 0-Ambience ",Aper,"%")
print(HHH,"samples 1-Hatchet ",Hper,"%")
print(CCC,"samples 2-Chainsaw ",Cper,"%")
print(VVV,"samples 3-Vehicle ",Vper,"%")

print ("Final Data Extracted Shape",meaningful.shape)
print ("Final Target Extracted Shape",meaningfultarget.shape)

dist = "TT-" +str(i) +" A-" +str(Aper) + " H-"+str(Hper)+" C-"+str(Cper) +" V-"+str(Vper)
print(dist)


# In[29]:


sampleNameHere = "NewTrue_Unique-Out-of-Sample"
#sampleNameHere = "NewInSample"


# # SAVE DATA AS NUMPY FILE

# In[30]:


from datetime import datetime
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime(" %d-%b-%Y-%H%M")
 
print('Current Tdatasetimestamp : ', timestampStr)


filesavename = sampleNameHere+"Mel-Log-"+dist+"-"+str(meaningful.shape[1]) +"-"+str(meaningful.shape[2])+timestampStr+"H.npy"

filesavename


# In[31]:



np.save("data"+filesavename, meaningful)
# save

np.save("target"+filesavename, meaningfultarget) # save


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[13]:


fs, signal = wav.read("TempDATA.WAV")
print(f)
print(fs,"kHz")
print("Channels",signal.shape)
if(np.array_equal(signal.shape, [220500, ])):
    print("1 Channel")
else:
    signal = signal[0:44100*5,0]
    print("2 Channels")
    
signal_preemphasized = speechpy.processing.preemphasis(signal, cof=0.98)
logenergy = speechpy.feature.lmfe(signal, sampling_frequency=fs, frame_length=para_fl, frame_stride=para_fl*0.5,
num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)


# In[11]:


logenergy[:][]


# In[31]:


dataset[1,:,:-1] = logenergy[:,:]


# In[27]:


dataset.shape


# In[28]:


import numpy as np
N = 2
a = np.random.rand(N,N)
b = np.zeros((N,N+1))
b[:,:-1] = a


# In[23]:


a


# In[24]:


b


# In[ ]:




