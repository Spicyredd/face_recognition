import pandas as pd
import os
import datetime

class TimeTable:
    def __init__(self, module_data = 'time_table.csv'):
        self.module_data = module_data
        if os.path.exists(module_data):
            self.data = pd.read_csv(module_data, index_col = 0)
        else:
            self.data = pd.DataFrame({
                }, index = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
            self.data.to_csv(module_data)
    
    def add_module_timing(self, module_day, module_name,  module_time):
        self.data.loc[module_day, module_name] = module_time
        self.data.to_csv(self.module_data)
        
    def get_module_timing(self, module_day, module_name):
        return self.data.loc[module_day, module_name]
    
    def remove_module_timing(self, module_day, module_name):
        self.data.loc[module_day, module_name] = ''
        self.data.to_csv(self.module_data)
        
    def get_info(self):
        return self.data.head()

    def get_current_weekday(self):
        current_weekday = datetime.datetime.now().strftime('%A')
        return current_weekday
    
    def get_current_time(self):
        current_time = datetime.datetime.now().strftime('%H:%M')
        return current_time

    def get_current_module(self, module_day, module_time):
        bool_series = self.data.loc[module_day].isin([module_time])
        if bool_series.any():
            module_name = bool_series[bool_series == True].index[0]
            return module_name
        else:
            return None

# time_table = TimeTable()
# # time_table.get_info()

# while True:
#     module_day = input('Enter the day of the week: ')
#     module_name = input('Enter the name of the module: ')
#     module_time = input('Enter the time of the module: ')
#     time_table.add_module_timing(module_day, module_name, module_time)
#     print("Added sucessfully.")
# time_table.get_info()
# while True:
#     weekday = time_table.get_current_weekday()
#     current_time = time_table.get_current_time()
# time_table.get_current_module('Monday', '11:00')+

