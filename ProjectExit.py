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
from datetime import datetime
import time
import RPi.GPIO as GPIO
from RpiMotorLib import RpiMotorLib
import tkinter as tk
#send to database and record if not exist
def sendDatabase(plate):
    
    x = datetime.now()
    x = x.strftime("%Y-%m-%d %H:%M:%S")
    
    #check if plate is already in database
    mycursor.execute(
        "SELECT * FROM plates WHERE license_plate = %s",
        (plate,)
    )
    check = mycursor.fetchone()
    # check if it is empty
    if not check:
        print ('Plate does not exist in the system.')
        #It should exist in the database since the vehicle entered the parking and exiting.
    else:
        print("Plate in the database")
        plate_id = check[0]
        print(plate)
        Exit(plate, x, plate_id)
        
        
def Exit(plate, x, plate_id):
    #get fee
    fee = Calculate(plate_id, x)
    print ("Fee: ")
    print(fee)
    status = "EXIT"
    #get user balance and deduct
    if (deductWallet(plate_id, fee)):
        print (x)
        sql = "UPDATE logs SET status=%s, exittime=%s, fee=%s, gate=%s WHERE plate_id=%s AND status='ENTER'"
        val = (status, x, fee, gatename, plate_id)
        mycursor.execute(sql, val)
        mydb.commit()
        print("vehicle exit successful")
        gateControl()
    else:
        print("No out >:(")
        time.sleep(5)
        
        
    
    
    
def Calculate(plate_id, x):
    mycursor.execute(
        "SELECT * FROM logs WHERE plate_id = %s AND status='ENTER'",
        (plate_id,)
    )
    check = mycursor.fetchone()
    if not check:
        print ('Entry log does not exist in the system')
        #It should exist in the database since the vehicle entered the parking and exiting.
    else:
        print("Log found in the database")
        fee = 0
        entrytime = check[1]
        exittime = datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        diff = abs(exittime - entrytime)
        days = diff.days
        secs = diff.seconds
        hours = secs // 3600
        minutes = (secs // 60 ) % 60
        #calculate fee (hardcoded for RM1 for 1 Hours)
        fee = fee + (days*24*1)
        fee = fee + (hours*1)
        if (minutes > 30):
            fee = fee + 1
        return fee
    
def deductWallet(plate_id, fee):
    mycursor.execute(
        "SELECT * FROM plates WHERE license_plate = %s",
        (plate,)
    )
    check = mycursor.fetchone()
    # check if it is empty
    if not check:
        print ('Plate not exist in the system')
        
    else:
        user_id = check[4]
        mycursor.execute(
        "SELECT * FROM users WHERE id = %s",
        (user_id,)
    )
    usercheck = mycursor.fetchone()
    if not usercheck:
        print("Plate is not registered to any account")
        
    else:
        balance = usercheck[10]
        if (fee > balance):
            print("Balance : ")
            print(balance)
            print("User balance not enough! Please add balance to your eWallet")
            usercheck = None
            return False
        else:
            
            balance = balance - fee
            print("New Balance : ")
            print(balance)
            sql = "UPDATE users SET balance = %s WHERE id = %s"
            var = (balance, user_id)
            mycursor.execute(sql, var)
            mydb.commit()
            usercheck = None
            return True
            
    
def gateControl():
    print("Opening Gate")
    gate.motor_run(GpioPins , .001, 128, False, False, "half", .05)
    time.sleep(3)
    print("Closing Gate")
    gate.motor_run(GpioPins , .001, 128, True, False, "half", .05)
    time.sleep(2)
    
    

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

#Get input from user for gate name
root= tk.Tk()

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
except:
    print("Failed to connect to database!")
    print("Please check db.ini file")
    detectstart = False
carcounter = 0

if (detectstart == True):
    while True:

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
                if (object_name == 'car'):
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
                        
                        print(carcounter)
                        if (carcounter > 5):
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
                                plate = data['results'][0]['plate'].upper()
                            sendDatabase(plate)
                            carcounter = 0
                    
                     
                
                
        # Draw framerate in corner of frame
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

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



