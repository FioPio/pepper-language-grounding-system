#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
           main.py

This file contains the code for the implementation of the language grounding
system using the robot Pepper.

First it records an audio that then it plays so the user knows what have been
recorded. After that, the audio is sent to the speech recognition server to
obtain a string that may be treated. After that, the robot takes a picture and
using the obtained query and the language grounding server it shows where the
object is on the image it took.
'''
__author__  = 'Ferriol Pey Comas [ferriol73pey@gmail.com]'
__version__ = 'v1.0'
__date__    = '13/05/2022'

from naoqi import ALProxy
from time import sleep
import numpy as np
from subprocess import call
import requests
import base64
import wave
from PIL import Image
import json
from io import BytesIO
import cv2

# This variable contains the Pepper robot IP
IP = "172.18.48.249"

# To use the leds of Pepper as indicators
leds = ALProxy("ALLeds", IP, 9559)

# Create led groups to control several leds at the same time
leds.createGroup("RightEye", ['RightFaceLed1','RightFaceLed2','RightFaceLed3',
                              'RightFaceLed4','RightFaceLed5','RightFaceLed6',
                              'RightFaceLed7','RightFaceLed8'])

leds.createGroup("LeftEye",  ['LeftFaceLed1', 'LeftFaceLed2', 'LeftFaceLed3',
                              'LeftFaceLed4', 'LeftFaceLed5', 'LeftFaceLed6',
                              'LeftFaceLed7', 'LeftFaceLed8'])

leds.createGroup("Eyes", ["RightEye", "LeftEye"])

###############################################################################
#
#  RECORDING THE AUDIO
#
###############################################################################

# To record audio on pepper
aurec = ALProxy("ALAudioRecorder", IP, 9559)
# Set eyes to color green, so the speaker knows pepper will listen
leds.fadeRGB('Eyes', 0, 1, 0, 0.5 )
sleep(1)
print("Starting recording ")
# Starts recording an audio un wave format just using the frontal mic and
# saves it to /home/nao/test.wav on the robot
aurec.startMicrophonesRecording("/home/nao/test.wav", "wav", 44100, [False, False, True, False])
# Waits 4 seconds before stoping the recording
sleep(4)
# Stops the recording
aurec.stopMicrophonesRecording()
print("audio recorded")
# Set the eyes to red to indicate the recording is done
leds.fadeRGB('Eyes', 1, 0, 0, 1 )
sleep(1)

# To play audios
player = ALProxy("ALAudioPlayer", IP, 9559)

# Loads the file pepper just records into pepper's memory so it may play it
fileID = player.loadFile("/home/nao/test.wav")

# Plays the recorded audio so the user knows what pepper has recorded
player.play(fileID)

# To be able to know the Pepper's battery charge
battery = ALProxy("ALBattery", IP, 9559)

# Displays the Pepper's battery charge
print(str(battery.getBatteryCharge()) + "% of battery remaining")


###############################################################################
#
#  COPYING FILE LOCALY
#
###############################################################################

# Uses the binary scp to copy the audio pepper just recorded into local storage
call(["scp","nao@"+IP+":/home/nao/test.wav", "."])

################################################################################
#
#  RECOGNIZING THE SPEECH
#
################################################################################

# The path to the file in which speech recognition is being applied
AUDIO_PATH ='test.wav'

# Encodes the audio to a base64 to send it using json
raw_data = base64.b64encode(open(AUDIO_PATH, "rb").read())

# Gets the framerate of the audio
fr = wave.open(AUDIO_PATH, "rb").getframerate()

#requests the transcription of the audio
res = requests.post('http://127.0.0.1:5002/api/recognize/01', json={"data":raw_data, "framerate":fr})

if res.ok:
    print(res.json())

predictQuery= res.json()['text']

################################################################################
#
#  APPLYING LANGUAGE GROUNDING
#
################################################################################

# To take pictures with the pepper camera
videoDevice = ALProxy('ALVideoDevice', IP, 9559)

# Subscribing to top camera
AL_kTopCamera = 0
AL_kQVGA = 1            # 320x240
AL_kBGRColorSpace = 13
captureDevice = videoDevice.subscribeCamera( "test", AL_kTopCamera, AL_kQVGA, AL_kBGRColorSpace, 10)

# Creating image variable to store the result
W = 320
H = 240
img = np.zeros((H, W, 3), np.uint8)

# Obtaining the raw image from the camera
res = videoDevice.getImageRemote(captureDevice)

# Unsubscribing from the camera so it may be used later
videoDevice.unsubscribe(captureDevice)

# Obrtaining the content of the raw image
if res == None:
    print 'cannot capture.'
elif res[6] == None:
    print 'no image data string.'
else:
    # translate value to mat
    values = map(ord, list(res[6]))
    i = 0
    for y in range(0, H):
        for x in range(0, W):
            img.itemset((y, x, 0), values[i + 0])
            img.itemset((y, x, 1), values[i + 1])
            img.itemset((y, x, 2), values[i + 2])
            i += 3

# Preparing the image to be send to the server-lg
PIL_image = Image.fromarray(np.uint8(img)).convert('RGB')

# Converting Pillow Image to bytes and then to base64
buffered = BytesIO()
PIL_image.save(buffered, format="JPEG")
img_byte = buffered.getvalue() # bytes
img_base64 = base64.b64encode(img_byte) #Base64-encoded bytes * not str

#It's still bytes so json.Convert to str to dumps(Because the json element does not support bytes type)
img_str = img_base64.decode('utf-8') # str

files = {
    "query":predictQuery,
    "img":img_str
    }

# Posting the image to the server with the query and obtaining the response
r = requests.post("http://127.0.0.1:5001", json=json.dumps(files))

print(r.json())

# Obtaining the box containing the requested object
box   = r.json()['box']

# Drawing a rectangle around the object
cv2.rectangle(img, pt1=(int(box[0]),int(box[1])), pt2=(int(box[2]),int(box[3])), color=(255,0,0), thickness=10)
# Showing the result
cv2.imshow("result", img)
# Waiting until a key is pressed
cv2.waitKey(0)
