import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import tkinter
import PIL.Image, PIL.ImageTk

vwidth = 640
vheight = 480

class App:
     def __init__(self, window, window_title, video_source=0):
         self.window = window
         self.window.title(window_title)
         self.video_source = video_source
 
         # open video source (by default this will try to open the computer webcam)
         self.vid = VideoStream(resolution=(640,480),framerate=30).start()
 
         # Create a canvas that can fit the above video source size
         self.canvas = tkinter.Canvas(window, width = vwidth, height = vheight)
         self.canvas.pack()
 
         # Button that lets the user take a snapshot
         self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
         self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)
 
         # After it is called once, the update method will be automatically called every delay milliseconds
         self.delay = 1
         self.update()
 
         self.window.mainloop()
         
     def snapshot(self):
         # Get a frame from the video source
         ret, frame = self.vid.get_frame()
 
         if ret:
             cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
 
     def update(self):
         # Get a frame from the video source
         frame = self.vid.read()
 
         
         self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
         self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
 
         self.window.after(self.delay, self.update)

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),rotation=90,framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True
        
        
App(tkinter.Tk(), "Tkinter and OpenCV")