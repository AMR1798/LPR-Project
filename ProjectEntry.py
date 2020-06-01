# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import requests
from pprint import pprint
import mysql.connector
import datetime
import time
import RPi.GPIO as GPIO
from RpiMotorLib import RpiMotorLib
import tkinter as tk
import threading
import base64
import io
from urllib.request import urlopen

#send to database and record if not exist
def sendDatabase(plate):
    
    x = datetime.datetime.now()
    x = x.strftime("%Y-%m-%d %H:%M:%S")
    
    #check if plate is already in database
    mycursor.execute(
        "SELECT * FROM plates WHERE license_plate = %s",
        (plate,)
    )
    check = mycursor.fetchone()
    # check if it is empty
    if not check:
        print ('It does not exist')
        sql = "INSERT INTO plates (license_plate, created_at, updated_at) VALUES (%s, %s, %s)"
        val = (plate, x, x)
        mycursor.execute(sql, val)
        mydb.commit()
        plate_id = mycursor.lastrowid
        entry(plate, x, plate_id)
    else:
        print("In the database")
        plate_id = check[0]
        entry(plate, x, plate_id)
        
        
def entry(plate, x, plate_id):
    #check if plate is already parked in the system
    mycursor.execute(
        "SELECT * FROM logs WHERE plate_id = %s AND status='ENTER'",
        (plate_id,)
    )
    check = mycursor.fetchone()
    if not check:
        #if does not exist
        print ('Entry log does not exist in the system')
        app.setTime(x)
        app.setGate(gatename)
        sql = "INSERT INTO logs(entry, status, created_at, updated_at, plate_id, fee, gate) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        val = (x, "ENTER", x, x, plate_id, "0.00", gatename)
        mycursor.execute(sql, val)
        mydb.commit()
        gateControl()
        print("vehicle entry successful")
    else:
        print("Vehicle already parked in the system")
        app.setMessage("Vehicle already parked in the system")
        time.sleep(5)
    
def gateControl():
    print("Opening Gate")
    app.setMessage("Opening Gate")
    gate.motor_run(GpioPins , .001, 128, False, False, "half", .05)
    time.sleep(3)
    app.setMessage("Closing Gate")
    print("Closing Gate")
    gate.motor_run(GpioPins , .001, 128, True, False, "half", .05)
    time.sleep(2)
    app.setPlate("")
    app.setTime("")
    app.setGate("")
    app.setMessage("")

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
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
        
class App(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.start()

    def callback(self):
        self.root.quit()

    def run(self):
        #specify image for parkey logo
        
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.callback)
        self.root.title("Parkey")
        self.canvas1 = tk.Canvas(self.root, width = 800, height = 600, bg='white')
        self.canvas1.pack()
        #load parkey logo
        url="https://i.imgur.com/GZKVNMe.png"
        image_byt = urlopen(url).read()
        image_b64 = base64.encodestring(image_byt)
        photo = tk.PhotoImage(data=image_b64)
        self.canvas1.create_image(250, 10, image=photo, anchor='nw')
        #connection status label
        self.connectLabel = tk.Label(self.root, text='Connecting', bg='white', fg="green")
        self.connectLabel.config(font=('helvetica', 15))
        self.canvas1.create_window(700, 580, window=self.connectLabel)
        #plate label
        self.plateLabel1 = tk.Label(self.root, text='Plate:', bg='white')
        self.plateLabel1.config(font=('helvetica', 50))
        self.canvas1.create_window(200, 200, window=self.plateLabel1)
        #plate number label
        self.plateLabel2 = tk.Label(self.root, text='', bg='white')
        self.plateLabel2.config(font=('helvetica', 45))
        self.canvas1.create_window(300, 165, window=self.plateLabel2, anchor='nw')
        #time label
        self.timeLabel1 = tk.Label(self.root, text='Time:', bg='white')
        self.timeLabel1.config(font=('helvetica', 30))
        self.canvas1.create_window(230, 300, window=self.timeLabel1)
        #time label show time
        self.timeLabel2 = tk.Label(self.root, text='', bg='white')
        self.timeLabel2.config(font=('helvetica', 30))
        self.canvas1.create_window(300, 275, window=self.timeLabel2, anchor='nw')
        #gate label
        self.gateLabel1 = tk.Label(self.root, text='Gate:', bg='white')
        self.gateLabel1.config(font=('helvetica', 30))
        self.canvas1.create_window(230, 380, window=self.gateLabel1)
        #gate label
        self.gateLabel2 = tk.Label(self.root, text='', bg='white')
        self.gateLabel2.config(font=('helvetica', 30))
        self.canvas1.create_window(300, 360, window=self.gateLabel2, anchor='nw')
        #message label
        self.messageLabel = tk.Label(self.root, text='', bg='white')
        self.messageLabel.config(font=('helvetica', 30))
        self.canvas1.create_window(400, 500, window=self.messageLabel, anchor='center')
        self.root.mainloop()
    def update(self):
        self.connectLabel['text'] = "Connected"   
    def setPlate(self, plate):
        self.plateLabel2['text'] = plate
    def setTime(self, time):
        self.timeLabel2['text'] = time
    def setGate(self, gate):
        self.gateLabel2['text'] = gate
    def setMessage(self, message):
        self.messageLabel['text'] = message



#Get input from user for gate name
root= tk.Tk()
root.title("Parkey: Initialization")

# Gets the requested values of the height and widht.
windowWidth = root.winfo_reqwidth()
windowHeight = root.winfo_reqheight()
 
# Gets both half the screen width/height and window width/height
positionRight = int(root.winfo_screenwidth()/2 - windowWidth/2)
positionDown = int(root.winfo_screenheight()/2 - windowHeight/2)
 
# Positions the window in the center of the page.
root.geometry("+{}+{}".format(positionRight, positionDown))
gatename = ""
canvas1 = tk.Canvas(root, width = 400, height = 300)
canvas1.pack()
label1 = tk.Label(root, text='Specify Gate Name')
label1.config(font=('helvetica', 14))
canvas1.create_window(200, 25, window=label1)
entry1 = tk.Entry (root) 
canvas1.create_window(200, 140, window=entry1)

def getName():  
    global gatename
    gatename = entry1.get()
    root.destroy()
    

button1 = tk.Button(text='Set Gate Name', command=getName)
canvas1.create_window(200, 180, window=button1)

root.mainloop()

#tensorflow settings
MODEL_NAME = "TFLite_model"
GRAPH_NAME = "detect.tflite"
LABELMAP_NAME = "labelmap.txt"
min_conf_threshold = float(0.5)
imW, imH = int(640), int(480)

#read db.ini for database connection settings
f = open("db.ini", "r")
x, host = f.readline().split('=')
x, username = f.readline().split('=')
x, password = f.readline().split('=')
x, database = f.readline().split('=')
password = password.rstrip()

app = App()
# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

 

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

#GPIO Pins for stepper motor (Simulate boom gate)
GpioPins = [17,18,27,22]
gate = RpiMotorLib.BYJMotor("Motor", "28BYJ")

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)
detectstart = True
try:
    print("Connecting to the database")
    config = {
          'user': username,
          'password': password,
          'host': host,
          'database': database,
          'raise_on_warnings': True,
          'autocommit' : True,
        }
    mydb = mysql.connector.connect(**config)
    print("Connection successful")
    mycursor = mydb.cursor()
except Exception as e:
    print(e)
    print("Failed to connect to database!")
    print("Please check db.ini file")
    detectstart = False
carcounter = 0

if(detectstart == True):
    app.update()
    while True:
        send = True
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        frame1 = videostream.read()

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame =  cv2.flip(frame1.copy(),-1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
        #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

        #car counter
        

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                if (object_name == "car"):
                    
                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))
                    
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                    
                    
                    
                
                    #Do something if found car / bus
                    if ((object_name == "car") or (object_name == "bus")):
                        carcounter = carcounter + 1
                        
                        app.setMessage("Car Detected")
                        if (carcounter > 5):
                            app.setMessage("Getting License Plate")
                            img_name = "car.jpg"
                            cv2.imwrite(img_name, frame)
                            regions = ['my'] # Change to your country
                            with open('car.jpg', 'rb') as fp:
                                response = requests.post(
                                    'https://api.platerecognizer.com/v1/plate-reader/',
                                    data=dict(regions=regions),  # Optional
                                    files=dict(upload=fp),
                                    headers={'Authorization': 'Token af581437289c4e9f3d6ccc38e82878638254e91c'})
                                data = response.json()
                                try:
                                    plate = data['results'][0]['plate'].upper()
                                    app.setMessage("License Plate Found")
                                except:
                                    app.setMessage("License Plate not found")
                                    print("License plate not found")
                                    time.sleep(3)
                                    send = False
                                if(send == True):
                                    app.setPlate(plate)
                                    sendDatabase(plate)
                            carcounter = 0
                    
                     
                
                
        # Draw framerate in corner of frame
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Car detector', frame)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

# Clean up
cv2.destroyAllWindows()
videostream.stop()



