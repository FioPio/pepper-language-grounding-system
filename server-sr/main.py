#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
           main.py

This script contains the implementation of a local server to do speech
recognition using a record of a voice in wav format. I is based on flask and
vosk.

'''
__author__  = 'Ferriol Pey Comas ( mrc20fpy ) [ferriol73pey@gmail.com]'
__version__ = 'v1.0'
__date__    = '07/04/2022'

from flask import Flask, request, jsonify # for the flask server
from vosk import Model, KaldiRecognizer, SetLogLevel # for the SR
import base64

SetLogLevel(-1)      # Deactivates the log information of vosk


app = Flask(__name__)

# Loading model
print('Building model')
model = Model("./model")

def recognize(data,fr):
    '''

    This function tries to convert into text some input audio.

    ==========
    Parameters
    ==========

       data :  aduio data
            It conains the data of the audio it has to be recognized.
       fr   : int
            It is the framerate of the audio.
    =======
    RETURNS
    =======

       result : json
           It returns the recognized sentence and the confidence, start
           time and end time for each recognized word.
    '''
    rec = KaldiRecognizer(model, fr)            # creates a recognizer for the selected model and the audio framerate
    rec.SetWords(True)                          # activates the output of the confidence, start time and end time for each word in the prediction
    rec.AcceptWaveform(data)                    # recognizes the audio
    return rec.Result()                         # returns the resulting json


@app.route('/api/recognize/<uuid>', methods=['GET', 'POST'])
def get_request(uuid):
    '''

    This function processes all the requests done to the url
    "/api/recognize/*" in order to return the speech recognition
    of the submited audio.
    ==========
    Parameters
    ==========
       uuid :  int
            It conains the uuid that the client selected and sent using
            the url.
    =======
    RETURNS
    =======
       result : json
           It returns the recognized sentence and the confidence, start
           time and end time for each recognized word.
    '''
    content = request.json                      # reads the json object from the request
    audio_raw = content['data']                 # obtains the audio data encoded in base64
    #decodes the audio
    decoded_audio = base64.b64decode(audio_raw) # decodes the base64 audio data to get it raw
    fr=content['framerate']                     # obtains the framerate of the audio from the request

    return recognize(decoded_audio,fr)          # returns the result of the recognition using the recognize function

if __name__ == '__main__':
    app.run(host= '0.0.0.0',debug=True)         # starts the application
