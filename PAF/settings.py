#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os

# SETTINGS
CUR_PATH = os.path.dirname(os.path.realpath(__file__)) #current path
SHAPE_PREDICTOR = os.path.join(CUR_PATH,"res","shape_predictor_68_face_landmarks.dat")
FACE_RECOGNITION_MODEL = os.path.join(CUR_PATH,"res","dlib_face_recognition_resnet_model_v1.dat")
DB_PATH = os.path.join(CUR_PATH,"data","paf.db")
#EVENTS_PATH = '/home/decadenza/Desktop/Python/20190611 Face GMT/Events/'
EVENTS_PATH = os.path.join(CUR_PATH,'..','Events')
# RECOGNITION TUNING
MAX_DISTANCE = 0.50                                             # Face recognition min threshold
OPENALPR_COUNTRY = "eu"                                         # Country for Plate Recognition
OPENALPR_MIN_CONFIDENCE = 0.5                                   # Plate confidence min threshold
#OPENALPR_CONF = '/etc/openalpr/openalpr.conf'                   # Openalpr configuration files
OPENALPR_CONF = '/etc/openalpr/openalpr.conf'                   # Openalpr configuration files
OPENALPR_RUNTIME_DATA = '/usr/share/openalpr/runtime_data'      # Openalpr configuration files
OPENALPR_ROTATIONS = [5,-5,10,-10,20,-20]                       # Image rotation angles (set to [] if not used)
