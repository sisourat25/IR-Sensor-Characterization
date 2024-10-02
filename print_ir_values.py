#
# Licensed under 3-Clause BSD license available in the License file. Copyright (c) 2021-2022 iRobot Corporation. All rights reserved.
#
from irobot_edu_sdk.backend.bluetooth import Bluetooth
from irobot_edu_sdk.robots import event, hand_over, Color, Robot, Root, Create3
from irobot_edu_sdk.music import Note

filename = input("Enter the filename\n")
if len(filename) < 0:
    filename = input("Enter the filename\n")

out_file = open(f"{filename}.csv", "a")
name = "CapstoneRobot1"
robot = Create3(Bluetooth(name))

out_file.write("left3, left2, left1, M, right1, right2, right3\n")
@event(robot.when_play)
async def play(robot):
    while True:
        sensors = (await robot.get_ir_proximity()).sensors
        sensor_string = str(sensors)
        # sensor_string = sensor_string[1:len(sensor_string) - 1] + "\n"
        sensor_string = sensor_string[1:len(sensor_string) - 1]
        print(sensor_string)
        # out_file.write(sensor_string)
        distance = 16

        await robot.navigate_to(0, distance)
        await robot.navigate_to(distance, distance)
        await robot.navigate_to(distance, 0)
        await robot.navigate_to(0, 0)
        print(sensor_string)
        
        distance = -distance

        await robot.navigate_to(0, distance)
        await robot.navigate_to(distance, distance)
        await robot.navigate_to(distance, 0)
        await robot.navigate_to(0, 0)

robot.play()