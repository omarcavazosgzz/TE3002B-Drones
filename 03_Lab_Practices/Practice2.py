import time
from djitellopy import Tello

tello = Tello()
tello.connect()      
tello.takeoff()
time.sleep(1)
tello.move_up(50)
time.sleep(1)
tello.move_forward(100)
time.sleep(1)
tello.move_right(100)
time.sleep(1)
tello.move_back(100)
time.sleep(1)
tello.move_left(100)
time.sleep(1)
tello.move_down(50)
time.sleep(1)
tello.land()


tello.takeoff()
time.sleep(1)
tello.move_up(50)
time.sleep(1)
tello.move_forward(100)
tello.rotate_clockwise(120)
time.sleep(1)
tello.move_forward(100)
tello.rotate_clockwise(120)
time.sleep(1)
tello.move_forward(100)
tello.rotate_clockwise(120)
time.sleep(1)
tello.move_down(50)
tello.land()



def fly_polygon(tello, sides, side_length=50, clockwise=True):
    if sides < 3:
        print("A shape needs at least 3 sides.")
        return

    if side_length < 20 or side_length > 500:
        print("Side length must be between 20 and 500 cm.")
        return

    angle = round(360 / sides)

    for _ in range(sides):
        tello.move_forward(side_length)
        if clockwise:
            tello.rotate_clockwise(angle)
        else:
            tello.rotate_counter_clockwise(angle)

try:
    sides = int(input("How many sides do you want? "))
    distance = int(input("How many cm per side? "))

    print("Battery:", tello.get_battery())

    tello.takeoff()
    time.sleep(2)

    fly_polygon(tello, sides, distance)

    time.sleep(1)
    tello.land()

except ValueError:
    print("Please enter whole numbers only.")
