# Acoustic Surveillance Intrusion Detecion system
# ASIDS Prototype 2020 for wildlife protection
Project ASIDS Development is a wide area intrusion detection surveillance system using audio to detect anomalous events that are activities of trespassers, poachers and illegal loggers. The system applies artifical intelligence to detect such illegal activities to be recognized as an intruder to assist wildlife protection efforts.    

# Introduction
ASIDS Introduction Video can be found in this link:
<br>https://youtu.be/cWPJTvhmXPM

<br>Project Milestone 1  
https://youtu.be/TN7kLvERBEc
<br>Project Milestone 2  
https://youtu.be/TN7kLvERBEc

# Hardware Used

<br>1.Raspberry Pi Zero Node + LoRA Pi-HatMulti Freq 
<br>2.Raspberry Pi 3B+ + Lora Gateway RAK 915
<br>Purchased from Cytron Technologies Sdn Bhd, Pulau Pinang, Malaysia

<br>USB Audio Device/Interface
<br>1.ATR2500 USB Microphone 
<br>2.BM-800 Condensor Mic + USB Audio Card
<br>3.Not Branded Mic + USB Audio Card
![ASIDS Node](https://github.com/ASIDS/Prototype/blob/master/ASIDS%20Node.jpg)
# Software 

Raspbian Buster (2020) OS and Python 3.

# Project Flow
## 1.Model Development
<br>
1. Acquire Audio Recordings
<br>
2. Feature Extraction MLE / LMFE
<br>
3. Feature Analysis 
<br>
4. Model Design
<br>
5. Model Training
<br>
6. Model Evaluation
<br>
7. Save and Embbed Model


## 2. Communication Setup
<br>
1. Raspberry Pi Master(Gateway) Setup
<br>
2. Raspberry Pi Slave(Node) Setup

Node Setup A
1-GPIO Signal Alarm method uses a LoRa Node Heltec MCU, Raspberry pi sendout digital signal 3.3v from GPIO to a Solid State Relay 3.3v. in time of alarm he node will continously send out message that has been pre-programeed to the Microcontroller.


Node Setup B
Lora Pi-Hat and Communicates with the Raspberry Pi over UART only using a total of 3 GPIO Pins
uses less than 50mA
<br>

## 3. Main Progam for A.S.I.Ds
<br>
1.Record Audio 
<br>
2.Preprocess Slice Audio, Convert to Single Channel 
<br>
3.Extract Features of each slice
<br>
4.Load Model -> Produce Inference
<br>
5.Compile Inference -> Create a Final Decision Filter
<br>
6.Sound Alarm -> LoRa Message -> Gateway
<br>
7.Sleep Until Next Routine, Redo Steps 1-7 each time device wake up

