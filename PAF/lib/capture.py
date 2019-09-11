#!/usr/bin/python3
# -*- coding: utf-8 -*-

import cv2
from threading import Thread

# Non blocking video capture!
class Capture:
    def __init__(self,source=0): # Source 0 is the default (usually webcam)
        if not isinstance(source, int) and source.isdigit():
            source = int(source)
        self.video_capture = cv2.VideoCapture(source)
        if not self.video_capture.isOpened():
            raise ValueError('Cannot open source!')
        self.flip = cv2.flip if source == 0 else lambda f, *a, **k: f # Flip around y axis only if is PC webcam
        self.running = False
        self.t=Thread(target=self.loop)
        self.t.daemon=True
        self.frame = None
        self.retrieve = self.video_capture.retrieve if isinstance(source,str) else self.video_capture.read
        self.grab = self.video_capture.grab if isinstance(source,str) else lambda *args: None # Grab only if it is not PC webcam, otherwise do nothing
        
    def start(self):
        self.running = True
        self.t.start()
    
    def stop(self):
        self.running = False
        self.t.join()
        return
        
    def loop(self):
        while(self.running):
            self.grab() # Keep grabbing frames...
            
    def get(self):
        ret, self.frame = self.retrieve() #capture frame-by-frame
        return self.flip(self.frame, 1)



# The following tries to elaborate EACH frame. Very slow! Do not use in production.
class CaptureEveryFrame:
    def __init__(self,source=0): # Source 0 is the default (usually webcam)
        self.video_capture = cv2.VideoCapture(source)
        self.running = False
        self.t=Thread(target=self.loop)
        self.t.daemon=True
        self.frame = None
        
    def start(self):
        self.running = True
        self.t.start()
    
    def stop(self):
        self.running = False
        self.t.join()
        return
        
    def loop(self):
        while(self.running):
            ret, self.frame = self.video_capture.read() #capture frame-by-frame
            
    def get(self):
        return self.frame
