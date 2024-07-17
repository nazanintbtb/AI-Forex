
import pandas as pd
import numpy as np
import csv

df = pd.read_csv('sorted_test_now.csv')
data = df["high_low"].tolist()
new_data=[]

new_data.append(data[0])
for i in range(1,len(data)):
    if(abs(data[i]-data[i-1])>=0.0002):
        new_data.append(data[i])


file_path = 'sorted_test_now_delete.csv'

# Write data to CSV
with open(file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write the column header
    writer.writerow(['validation'])
    
    # Write the list elements as rows
    for item in new_data:
        writer.writerow([item])

    
    



