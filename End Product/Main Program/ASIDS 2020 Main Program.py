from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import os
import wave, sys, pyaudio
import contextlib
from pydub import AudioSegment
import scipy.io.wavfile as wav
import speechpy
import wave
import numpy as np
from sklearn.metrics import classification_report

# load json and create model
root ="C:/Users/Another/CNN/Batch Trainning/Final Model/" 
NAME = root + "32 64-Conv 256 -Dense Batch 1576410952"

json_file = open(NAME+"model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(NAME+"model.h5")
print("Loaded model from disk")
model =loaded_model


def recordNow(): 
    import pyaudio
    import wave

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 30
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
    
def decode(datum):
    return np.argmax(datum)


for z in range(1):
	recordNow()

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

	#DEPENDS ON THE AMOUNT OF TOTAL EXTRACTIONS JUST PUT 1000 TO BE SAFE THEN REDUCE
	dataset = np.ndarray(shape=(100,98,40))

	#mysoundhere= "Sample\Car 30m.WAV"
	#mysoundhere= "Sample\Car 50m.WAV"
	#mysoundhere= "Sample\Chainsaw 50m.WAV"
	#mysoundhere= "Sample\Hachet 30m.WAV"
	#mysoundhere= "Sample\Machete 30m.WAV"
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
			num_filters=40, fft_length=1024, low_frequency=0, high_frequency=None)
			
			dataset[i] = logenergy
			#print("Data ",logenergy.shape)
			i += 1   
			
	dataset = dataset[0:i]


	inputData = dataset.reshape(-1,dataset.shape[1],dataset.shape[2],1)


	test_predictions = model.predict_classes(inputData)
	rawPred = model.predict(inputData)

	results = np.ndarray(shape=(inputData.shape[0],))


	xx = np.round(rawPred,3)

	arr2D = xx
	maxInRows = np.amax(arr2D, axis=1)
	t0 = 0
	t1 = 0
	t2 = 0
	t3 = 0

	for i in range(xx.shape[0]):
		#print(np.amax(xx[i]))
		result = np.where(xx[i] == maxInRows[i])
		if(result[0] == 0):
			predicted = "Ambience"
			t0 += maxInRows[i]
		elif(result[0] == 1):
			t1 += maxInRows[i]
			predicted = "Axe"
		elif(result[0] == 2):
			t2 += maxInRows[i]
			predicted = "Chainsaw"
		elif(result[0] == 3):
			t3 += maxInRows[i]
			predicted = "Vehicle"
			

		print("Segment "+str(i+1), ": ",predicted,round(maxInRows[i]*100,2),"%",)
	
	
	
	

	encoded_data = test_predictions

	for i in range(encoded_data.shape[0]):
		datum = encoded_data[i]
		#print('index: %d' % i)
		#print('encoded datum: %s' % datum)
		decoded_datum = decode(encoded_data[i])
		results[i] = decoded_datum
		#print('decoded datum: %s' % decoded_datum)
		#print()
	rawPred.shape
	rawPred
	confidence = 0.70

	rr = rawPred
	rr = np.round(rr,2)
	rr = np.where(rr > confidence, 1, 0)
	rr[1] == [0,0,0,0]

	#Filter 0,0,0,0 into Ambience Class- 0
	idx = 0
	for i in (rr):
		
		if((i == [0,0,0,0]).all()):
			#print(i)
			rr[idx] = [1,0,0,0]
			#print("Low Conficdence " , idx)
		idx += 1
		
	## DECODE [0,0,0,1] AS 3 Class Numbers  
	encoded_data = rr

	for i in range(encoded_data.shape[0]):
		datum = encoded_data[i]
		#print('index: %d' % i)
		#print('encoded datum: %s' % datum)
		decoded_datum = decode(encoded_data[i])
		results[i] = decoded_datum
		#print('decoded datum: %s' % decoded_datum)
		#print(results)
		   
	results 
	x =results
	ambience = np.sum(x == 0)
	axe = np.sum(x == 1)
	chainsaw = np.sum(x == 2)
	vehicle = np.sum(x == 3)
	print("Analyzed 30 Seconds -", x.shape[0] ,"Segments")
	total = x.shape[0]
	print("Decision Threshold 80%")
	print("Ambience     :",round((ambience/total)*100,2),"%")
	print("Axe Activity :",round((axe/total)*100,2),"%")
	print("Chainsaw     :",round((chainsaw/total)*100,2),"%")
	print("Vehicle      :",round((vehicle/total)*100,2),"%")
	
	x =test_predictions
	ambience = np.sum(x == 0)
	axe = np.sum(x == 1)
	chainsaw = np.sum(x == 2)
	vehicle = np.sum(x == 3)
	print("Analyzed 30 Seconds -", x.shape[0] ,"Segments")
	total = x.shape[0]
	print("Decision None")
	print("Ambience     :",round((ambience/total)*100,2),"%")
	print("Axe Activity :",round((axe/total)*100,2),"%")
	print("Chainsaw     :",round((chainsaw/total)*100,2),"%")
	print("Vehicle      :",round((vehicle/total)*100,2),"%")
		
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
