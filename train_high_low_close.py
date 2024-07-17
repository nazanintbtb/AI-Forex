import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def scale_list(numbers):
    min_num = min(numbers)
    max_num = max(numbers)
    scaled_numbers = []

    for num in numbers:
        scaled_num = 1 + (num - min_num) * (2 - 1) / (max_num - min_num)
        scaled_numbers.append(scaled_num)

    return scaled_numbers


# Read the CSV file
df = pd.read_csv('train_data.csv')
High_low_seq = df["validation"].tolist()

window_size = 1200
stride = 1

inputs_high = []
target_high = []

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
        t=2+(diff*(abs(input_norm[1]-input_norm[2])/abs( my_input[1]-my_input[2])))
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
        target_high.append([t])
        
        
        # print(tt)
        # print(diff)
        # print(t)

       

    else:
        # print("---------------------_------------------------------")
        tt.append(t)
        tt_norm = scale_list(tt)
        t=tt_norm[-1]
        target_high.append([t])
        
        # print(tt)
        # print(tt_norm)


    # if(i==245):
    #     break

        




    # target_high.append([t])

inputs_high = np.array(inputs_high)
target_high = np.array(target_high)

# # Normalize the inputs and targets
# scaler = MinMaxScaler()
# inputs_high = scaler.fit_transform(inputs_high)
# target_high = scaler.fit_transform(target_high)

# Reshape the inputs and targets
inputs = inputs_high.reshape((inputs_high.shape[0], window_size, 1))
target = target_high.reshape((target_high.shape[0], 1, 1))

# Split the data into training and testing sets
train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs, target, test_size=0.1)

# Define the input sequence length and the number of features
seq_length = train_inputs.shape[1]
num_features = train_inputs.shape[2]

# Define the encoder LSTM layer as bidirectional
encoder_inputs = tf.keras.Input(shape=(seq_length, num_features))
encoder_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, return_state=True))
encoder_outputs, forward_state_h, forward_state_c, backward_state_h, backward_state_c = encoder_lstm(encoder_inputs)
state_h = tf.keras.layers.Concatenate()([forward_state_h, backward_state_h])
state_c = tf.keras.layers.Concatenate()([forward_state_c, backward_state_c])
encoder_states = [state_h, state_c]

# Define the decoder LSTM layer
decoder_inputs = tf.keras.Input(shape=(1, num_features))
decoder_lstm = tf.keras.layers.LSTM(512, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(num_features, activation='linear')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Define callbacks
checkpoint = ModelCheckpoint('finance.h5', monitor='val_loss', save_best_only=True, save_weights_only=False, mode='min', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1)

# Train the model
model.fit([train_inputs, train_inputs[:, 1199:1200, :]], train_targets, epochs=300, batch_size=128,
          validation_data=([test_inputs, test_inputs[:, 1199:1200, :]], test_targets), callbacks=[checkpoint, early_stopping])