from interface import TimeTable
import time
import timer



time_table = TimeTable()

while True:
    current_time = time_table.get_current_time()
    current_weekday = time_table.get_current_weekday()
    current_module = time_table.get_current_module(current_weekday, current_time)
    print(current_module)