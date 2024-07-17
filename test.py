import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import csv
from sklearn.preprocessing import MinMaxScaler


def scale_list(numbers):
    min_num = min(numbers)
    max_num = max(numbers)
    scaled_numbers = []

    for num in numbers:
        scaled_num = 1 + (num - min_num) * (2 - 1) / (max_num - min_num)
        scaled_numbers.append(scaled_num)

    return scaled_numbers


df = pd.read_csv('test_data_4.csv')
High_low_seq=df["validation"].tolist()

window_size = 1200
stride = 1

inputs_high = []
target_high=[]

bias=[]

for i in range(0, len(High_low_seq) - window_size, stride):
    my_input = High_low_seq[i:i + window_size]

    # print("input")
    # print(my_input[200:210])
    tt=[]

    minpos = my_input.index(min(my_input))
    tt.append(my_input[minpos])
    maxpos = my_input.index(max(my_input))

    tt.append(my_input[maxpos])
    # scaler = MinMaxScaler()
    input_norm = scale_list(my_input)
    # print("input norm")
    # print(input_norm[200:210])
    
    inputs_high.append(input_norm)
    
    t = High_low_seq[i + window_size]
    

    if(t>=my_input[maxpos]):

        # print("---------------------t>------------------------------")
        diff=t-my_input[maxpos]
        # tt.append(diff)
        # tt_norm = scale_list([0,1,diff])
        t=2+(diff*(abs(input_norm[0]-input_norm[1])/abs( my_input[0]-my_input[1])))
        bias.append(abs(input_norm[0]-input_norm[1])/abs( my_input[0]-my_input[1]))

        target_high.append([t])

        # print(tt)
        # print(diff)
        # print(t)

        

    elif(t<=my_input[minpos]):
        # print("---------------------t<------------------------------")
        diff=my_input[minpos]-t
        # tt.append(diff)
        # tt_norm = scale_list([0,1,diff])
        t=1-(diff*(abs(input_norm[0]-input_norm[1])/abs( my_input[0]-my_input[1])))
        bias.append(abs(input_norm[0]-input_norm[1])/abs( my_input[0]-my_input[1]))
        target_high.append([t])
        
        
        # print(tt)
        # print(diff)
        # print(t)

       

    else:
        # print("---------------------_------------------------------")
        bias.append(abs(input_norm[0]-input_norm[1])/abs( my_input[0]-my_input[1]))
        tt.append(t)
        tt_norm = scale_list(tt)
        t=tt_norm[-1]
        target_high.append([t])

   


inputs_high=np.array(inputs_high)
target_high=np.array(target_high)

inputs = inputs_high.reshape((inputs_high.shape[0], 1200, 1))
target = target_high.reshape((target_high.shape[0], 1, 1))





model = load_model('finance.h5')

# Use the loaded model for prediction
predictions = model.predict([inputs, inputs[:, 1199:1200, :]])


  

predictions = np.reshape(predictions, (len(predictions),))

target = np.reshape(target, (len(target),))
# target = np.append(target, 0)

pr=(predictions[-1]-target[-2])/bias[-1]
# print(predictions[-1],target[-2],bias[-1])

print("prrrrrrrrrrrrrrr=")
print(pr)

data = [{'predict': p, 'target': t, "bias": b } for p, t,b in zip(predictions, target,bias)]

# Specify the CSV file path
csv_file_path = 'result_1200.csv'

# Write data to CSV file
with open(csv_file_path, 'w', newline='') as file:
    fieldnames = ['predict', 'target', "bias"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    
    writer.writeheader()  # Write column names
    writer.writerows(data)  # Write data rows

print('CSV file has been successfully created.')

