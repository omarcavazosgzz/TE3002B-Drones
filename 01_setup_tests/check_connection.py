from djitellopy import Tello

tello = Tello()
tello.connect()

print("Battery:", tello.get_battery())
print("SDK version:", tello.query_sdk_version())

tello.end()
