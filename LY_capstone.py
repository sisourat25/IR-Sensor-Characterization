
from irobot_edu_sdk.backend.bluetooth import Bluetooth
from irobot_edu_sdk.robots import event, hand_over, Color, Robot, Root, Create3
from irobot_edu_sdk.music import Note
import time

name = "CapstoneRobot1"

robot = Create3(Bluetooth(name))
sensor_data = [0] * 7
@event(robot.when_bumped, [True, False])
async def bumped(robot):
    print('Left bump sensor hit')
    
@event(robot.when_bumped, [False, True])
async def bumped(robot):
    print('Right bump sensor hit')
    print(sensor_data)
@event(robot.when_play)
async def play(robot):
    while True:
        sensors = (await robot.get_ir_proximity()).sensors
        sensor_data[0] = sensors[0]
        sensor_data[1] = sensors[1]
        sensor_data[2] = sensors[2]
        sensor_data[3] = sensors[3]
        sensor_data[4] = sensors[4]
        sensor_data[5] = sensors[5]
        print("Ir sensor input:", sensors)