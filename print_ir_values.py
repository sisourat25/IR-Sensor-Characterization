#
# Licensed under 3-Clause BSD license available in the License file. Copyright (c) 2021-2022 iRobot Corporation. All rights reserved.
#
from irobot_edu_sdk.backend.bluetooth import Bluetooth
from irobot_edu_sdk.robots import event, Create3
import asyncio
import os

task = None
cancelled = False
filename = input("Enter the filename\n")
if len(filename) < 0:
    filename = input("Enter the filename\n")
    
"""
Example Usage:
Enter the grid squares the object is in
a1,a2,a3,b3 <-- User Input
['a1', 'a2', 'a3', 'b3']
"""
zones = input("Enter the grid squares the object is in\n").split(",")
filename = f"{filename}.csv"

out_file = open(filename, "a")
if not os.path.isfile(filename):
    out_file.write("zones, left3, left2, left1, M, right1, right2, right3\n")

name = "CapstoneRobot1"
robot = Create3(Bluetooth(name))

@event(robot.when_play)
async def play(robot):
    global cancelled
    # task = asyncio.create_task(move(robot))
    while True:
        sensors = (await robot.get_ir_proximity()).sensors
        sensor_string = str(sensors)
        # sensor_string = sensor_string[1:len(sensor_string) - 1] + "\n"
        sensor_string = f"{zones},{sensor_string[1:len(sensor_string) - 1]}"
        print(sensor_string)
        # out_file.write(sensor_string)

async def move(robot):
        curr = 0
        distance = 100
        await robot.set_wheel_speeds(5, -5)
        while True:
            await robot.navigate_to(0, distance)
            await robot.navigate_to(distance, 0)
            curr += 1
        # await robot.navigate_to(distance, distance)
        # await robot.navigate_to(distance, 0)
        # await robot.navigate_to(0, 0)
        
        # distance = -distance

        # await robot.navigate_to(0, distance)
        # await robot.navigate_to(distance, distance)
        # await robot.navigate_to(distance, 0)
        # await robot.navigate_to(0, 0)    
robot.play()