from data_control import TimeTable
import time
from camera_control import camera_control

timed_module = None

while True:
    time_table = TimeTable()
    current_time = time_table.get_current_time()
    current_weekday = time_table.get_current_weekday()
    current_module = time_table.get_current_module(current_weekday, current_time)
    if current_module != timed_module and current_module != None:
        timed_module = current_module
        for i in range(3):
            camera_control()
            print(f"Current module: {current_module}")
            time.sleep(60)
    time.sleep(10)