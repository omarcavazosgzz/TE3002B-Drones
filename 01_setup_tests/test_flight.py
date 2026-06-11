import time
from djitellopy import Tello

tello = Tello()
tello.connect()

print("Battery:", tello.query_battery())
tello.takeoff()
time.sleep(5)
tello.land()