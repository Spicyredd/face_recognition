import pandas as pd
data = pd.DataFrame({
    'CMP 1234': ['10:15', '10:35', '09:40', '10:25', '08:55', '10:25', '08:45'],
    }, index = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
# print(data['CMP 1234']['Monday'])
# data.loc['Sunday', 'CMP 1234'] = '10:25'
# print(data.loc['Sunday', 'CMP 1234'])
print(data['CMP 1234']['Monday'])
# print(data.head())
# data.head()