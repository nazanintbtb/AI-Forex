import pandas as pd
import numpy as np
import csv

def closest_number(target, numbers):
    numbers = list(numbers)  # Convert RangeIndex to a list
    closest_val = min(numbers, key=lambda x: abs(x - target))
    closest_idx = numbers.index(closest_val)
    return closest_idx


def findelement(search_element,my_list):
    ind=""
    for index, element in enumerate(my_list):
        if element == search_element:
            ind=index
            break        
    return ind

df = pd.read_csv('apply_sort.csv')
High_15=df["High_10"].tolist() 
Low_15=df["Low_10"].tolist() 
date_15=df["date_10"].tolist() 



High_4=df["High_1"].tolist() 
Low_4=df["Low_1"].tolist() 
date_4=df["date_1"].tolist() 
Close=df["Close_1"].tolist() 

order=[]

for i in range(128057-2):
    print(i)
    date4=date_4[i]
    date4_1=date_4[i+1]
    high4=High_4[i]
    low4=Low_4[i]
    close=Close[i]

    index1=findelement(date4,date_15)
    index2=findelement(date4_1,date_15)
    print("date4_1",date4_1)
    print("date4",date4)
    print("------------------------------------------------------")
    

    high_list1=High_15[index1:index2]
    low_list1=Low_15[index1:index2]

    
    # high_list1=High_1[i*4:((i+1)*4)+4]
    # low_list1=Low_1[i*4:((i+1)*4)+4]

    

    low_index = closest_number(low4, low_list1)
    high_index = closest_number(high4, high_list1)

    if(high_index>low_index):
        order.append(low4)
      
        order.append(high4)
        order.append(close)
      
    else:
        order.append(high4)
       
        order.append(low4)
        order.append(close)
    

    


def save_list_to_csv(data_list, file_name):
    # Define the column headers
    fieldnames = ['high_low']

    # Open the CSV file in write mode
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the column headers
        writer.writeheader()

        # Write the data
        for item in data_list:
            writer.writerow({'high_low': item})

# Example usage

file_name = 'sorted_test_now.csv'

save_list_to_csv(order, file_name)



