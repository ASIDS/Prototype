#============================================================================================================
#|||||||||||||||||||||||||||||||||| IMPORT LIBRARIES|||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#============================================================================================================ 
import wave, sys, pyaudio
import contextlib
from pydub import AudioSegment
import scipy.io.wavfile as wav
import pickle; print("pickle", sys.version)
import timeit; print("timeit", sys.version)
import contextlib; print("contextlib", sys.version)
import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy as np; print("NumPy", np.__version__)
import scipy; print("SciPy", scipy.__version__)
import sklearn; print("Scikit-Learn", sklearn.__version__)
import speechpy
print("-------------------  ALL IMPORTS COMPLETE ------------------------")

#============================================================================================================
#|||||||||||||||||||||||||||||||||| Define Functions|||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#============================================================================================================ 
def OneHotEncode(x,classAmount):
    
    from numpy import argmax
    # integer encode input data
    integer_encoded = list(np.floor(x).astype(int))
    # one hot encode
    onehot_encoded = list()
    for value in integer_encoded:
        temp = [0 for _ in range(classAmount)]
        temp[value] = 1
        onehot_encoded.append(temp)
    return np.array(onehot_encoded)
# invert encoding
def recordNow(): 
    import pyaudio
    import wave

    CHUNK = 8192
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = RecTime
    WAVE_OUTPUT_FILENAME = "sound.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Listening")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Done")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    
def flattenFeature(x):
    flats = []
    print(len(x))
    for i in range(len(x)):
        flat = np.ndarray.flatten(x[i,:,:])
        flats.append(flat)
    return np.array(flats)



def decode(datum):
    return np.argmax(datum)


def PersonalFilter(ZZ,CONF0,CONF1,CONF2,CONF3):
    for i in ZZ:
        if(i[0] > CONF0):
            i[0] = 1
        else:
            i[0] = 0

    for i in ZZ:
        if(i[1] > CONF1):
            i[1] = 1
        else:
            i[1] = 0

    for i in ZZ:
        if(i[2] > CONF2):
            i[2] = 1
        else:
            i[2] = 0

    for i in ZZ:
        if(i[3] > CONF3):
            i[3] = 1
        else:
            i[3] = 0
    idx = 0
    for i in ZZ:
        if((i == [0,0,0,0]).all()):
            ZZ[idx] = [1,0,0,0]
            #print("Low Conficdence == NULL" , idx)
        elif(ZZ[idx][0] == 1):
            ZZ[idx] = [1,0,0,0]
        idx += 1
    return ZZ
##DECODE BINARY TARGET TO NUMERIC TARGET 0 1 0 0 TO 2###


def decodeRows(encoded_data):
    temp = []
    for i in range(encoded_data.shape[0]):
        datum = encoded_data[i]
        #print('index: %d' % i)
        #print('encoded datum: %s' % datum)
        decoded_datum = decode(encoded_data[i])
        temp.append(decoded_datum)
    return np.array(temp)
    #print('decoded datum: %s' % decoded_datum)
    #print()
def decode(datum):
    return np.argmax(datum)

def OneHotEncode(x,classAmount):
    
    from numpy import argmax
    # integer encode input data
    integer_encoded = list(np.floor(x).astype(int))
    # one hot encode
    onehot_encoded = list()
    for value in integer_encoded:
        temp = [0 for _ in range(classAmount)]
        temp[value] = 1
        onehot_encoded.append(temp)
    return np.array(onehot_encoded)
# invert encoding
#============================================================================================================
#|||||||||||||||||||||||||||||||||| IMPORT MODEL ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#============================================================================================================ 
print("-------------------    Loading Model   ------------------------")
ModelLoading = timeit.default_timer()
filename = 'RandomForest100-ALLDATA.sav'
loaded_model = pickle.load(open(filename, 'rb'))
ModelLoadingDone = timeit.default_timer()
print('Model Loading Time: ', ModelLoadingDone - ModelLoading)
RecTime = 15
MonitorIteration = 1
Threatalert = False
GlobalThresh = 50
debug = True
#debug = False
refineResults = True
#refineResults = False
#Debug Mode Skip Listening and uses the sound.wav file through testing
#============================================================================================================
#|||||||||||||||||||||||||||||||||| RECORD AND EXTRACT FEATURE |||||||||||||||||||||||||||||||||||||||||||||
#============================================================================================================ 


for monitor in range(MonitorIteration):
	Threatalert = False
	if not debug:
		 recordNow()
	#Calculate Time to Run START
	start = timeit.default_timer()
	cutSeconds = 5
	slideSeconds = 0.5
	shift = int(cutSeconds/slideSeconds) - 1
	#print("Audio Shift seconds",slideSeconds)
	#print("Audio Shifts Segments",shift)
	if(shift == 0):
		shift = 1
	i =0
	para_fl = 0.1
	para_ovlp = 0.8
	#DEPENDS ON THE AMOUNT OF TOTAL EXTRACTIONS JUST PUT 100 TO BE SAFE THEN REDUCE
	dataset = np.ndarray(shape=(100,98,40))

	mysoundhere= "sound.wav"
	for z in range(shift): 
		f = mysoundhere
		fname = mysoundhere
		with contextlib.closing(wave.open(fname,'r')) as faa:
			frames = faa.getnframes()
			rate = faa.getframerate()
			duration = frames / float(rate)
		#print("Reading File : ",fname,"With Target Label",1)
		#print('SR',rate)
		#print('duration',duration)
		segs =int(duration/cutSeconds)
		#print('Segments',segs)    
		y = AudioSegment.from_wav(f)
		ShiftAudio = y[(slideSeconds*1000)*z:(cutSeconds*1000)*(segs)]
		segs = segs-1
		for sss in range(segs): 
			newAudio = ShiftAudio[(cutSeconds*1000)*sss:(cutSeconds*1000)*(sss+1)]

			##COMMENT OUT SAVE FILE TO TEST RUN 
			newAudio.export("TempDATA.WAV", format="wav")

			fs, signal = wav.read("TempDATA.WAV")
			#print(f)
			#print(fs,"kHz")
			#print("Channels",signal.shape)
			if(np.array_equal(signal.shape, [220500, ])):
				zzz=0
				#print("1 Channel")
			else:
				signal = signal[0:44100*5,0]
				#print("2 Channels")

			signal_preemphasized = speechpy.processing.preemphasis(signal, cof=0.98)
			logenergy = speechpy.feature.lmfe(signal, sampling_frequency=fs, frame_length=para_fl, frame_stride=para_fl*0.5,
			num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
			
			dataset[i] = logenergy
			#print("Data ",logenergy.shape)
			i += 1
			
	dataset = dataset[0:i]

#============================================================================================================
#|||||||||||||||||||||||||||||||||| FLATTEN FEATURE AND PREDICT||||||||||||||||||||||||||||||||||||||||||||||
#============================================================================================================    
print('Features : ',dataset.shape,' x ',i,' Segments')
print('Features Extraction Time :',timeit.default_timer()-start)

X = flattenFeature(dataset)
print('Flatten Features Shape:', X.shape)
start = timeit.default_timer()
prediction = loaded_model.predict_proba(X)
print('Prediction Time :',timeit.default_timer()-start)
#============================================================================================================
#|||||||||||||||||||||||||||||||||| REFINE OUTPUT |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#============================================================================================================
start = timeit.default_timer()
Results = PersonalFilter(np.array(prediction),0.3,0.7,0.5,0.5)
print('Refining Time :',timeit.default_timer()-start)
print(prediction)
print(Results)
#============================================================================================================
#|||||||||||||||||||||||||||||||||| REFINE OUTPUT |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#============================================================================================================
if(refineResults):
	xx = prediction
	results = xx
	rawPred = xx
	arr2D = xx
	maxInRows = np.amax(arr2D, axis=1)
	t0 = 0
	t1 = 0
	t2 = 0
	t3 = 0

	for i in range(xx.shape[0]):
		#print(np.amax(xx[i]))
		result = np.where(xx[i] == maxInRows[i])
		if(result[0][0] == 0):
			predicted = "Ambience"
			t0 += maxInRows[i]
		elif(result[0] == 1):
			t1 += maxInRows[i]
			predicted = "Hatchet"
		elif(result[0] == 2):
			t2 += maxInRows[i]
			predicted = "Chainsaw"
		elif(result[0] == 3):
			t3 += maxInRows[i]
			predicted = "Vehicle"
			

		print("Segment "+str(i+1), ": ",predicted,round(maxInRows[i]*100,2),"%",)    
		preditedData = "\nSegment "+str(i+1)+": "+ predicted +" - "+str(round(maxInRows[i]*100,2))+"%"


	tglobal = t0 + t1 + t2 + t3
	print("------------ Global Average ------------")    
	print("Ambience ",round((t0/tglobal)*100,2),"%")
	print("Axe ",round((t1/tglobal)*100,2),"%")
	print("Chainsaw ",round((t2/tglobal)*100,2),"%")
	print("Vehicle ",round((t3/tglobal)*100,2),"%")
		
		
	print("------------ Global Confidence -------------")    
	print("Ambience ",round((t0/36)*100,2),"%")
	print("Axe ",round((t1/36)*100,2),"%")
	print("Chainsaw ",round((t2/36)*100,2),"%")
	print("Vehicle ",round((t3/36)*100,2),"%")

	#Calculate Time to Run END
	stop = timeit.default_timer()
	print('Recording Time: ',RecTime)
	print('Prediction Time: ', stop - start)

	if(finalAxeDecision > GlobalThresh):
		print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@AxeThreat INTRUDER ALERT@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
		LogAlert = "\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@AxeThreatINTRUDER ALERT@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
		Threatalert = True
	elif(finalChainsawDecision > GlobalThresh):
		print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ChaThreat INTRUDER ALERT@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
		LogAlert = "\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ChaThreat INTRUDER ALERT@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" 
		Threatalert = True            
	elif(finalVehicleDecision > GlobalThresh):
		print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@VehThreatINTRUDER ALERT@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
		LogAlert = "\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@VehThreat INTRUDER ALERT@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" 
		Threatalert = True        
	else:
		print("Nothing To See Here")
		LogAlert = "\n===================================Nothing To See Here=========================="  
		Threatalert = False     
		
	# IF ALARM TRUE --> TURN ON ALARM THROUGH GPIO PIN 18 comment out FALSE ON PI 
	Threatalert = False
	if(Threatalert):

		import RPi.GPIO as GPIO
		import time

		GPIO.setmode(GPIO.BCM)
		GPIO.setup(18,GPIO.OUT)

		GPIO.output(18,GPIO.HIGH)
		#Sleep 5 Seconds
		time.sleep(2)
		GPIO.output(18,GPIO.LOW)
		print("Threats Around, Commence to Alarm")
		


#============================================================================================================
#|||||||||||||||||||||||||||||||||| REFINE OUTPUT |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#============================================================================================================