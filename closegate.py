import RPi.GPIO as GPIO
from RpiMotorLib import RpiMotorLib
#GPIO Pins for stepper motor (Simulate boom gate)
GpioPins = [17,18,27,22]
gate = RpiMotorLib.BYJMotor("Motor", "28BYJ")


gate.motor_run(GpioPins , .001, 128, True, False, "half", .05)