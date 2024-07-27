import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model


df = pd.read_csv('data.csv')
High_seq=df["High"]
Low_seq=df["Low"]
close_seq=df["Close"]

window_size = 240
stride = 1

inputs_high = []
target_high=[]
for i in range(0, len(High_seq) - window_size  , stride):
    input = High_seq[i:i+window_size]

    inputs_high.append(input)
    t=High_seq[i+window_size]
    target_high.append([t])

inputs_low = []
target_low=[]
for i in range(0, len(Low_seq) - window_size  , stride):
    input = Low_seq[i:i+window_size]

    inputs_low.append(input)
    t=Low_seq[i+window_size]
    target_low.append([t])


inputs_close = []
target_close=[]
for i in range(0, len(close_seq) - window_size  , stride):
    input = close_seq[i:i+window_size]

    inputs_close.append(input)
    t=close_seq[i+window_size]
    target_close.append([t])


inputs_high=np.array(inputs_high)
target_high=np.array(target_high)
print(inputs_high.shape)
print(target_high.shape)

inputs_low=np.array(inputs_low)
target_low=np.array(target_low)
print(inputs_low.shape)
print(target_low.shape)

inputs_close=np.array(inputs_close)
target_close=np.array(target_close)
print(inputs_close.shape)
print(target_close.shape)

inputs_high = inputs_high.reshape((inputs_high.shape[0], 240, 1))
target_high = target_high.reshape((target_high.shape[0], 1, 1))

inputs_low = inputs_low.reshape((inputs_low.shape[0], 240, 1))
target_low = target_low.reshape((target_low.shape[0], 1, 1))

inputs_close = inputs_close.reshape((inputs_low.shape[0], 240, 1))
target_close = target_close.reshape((target_low.shape[0], 1, 1))


inputs = np.concatenate((inputs_high, inputs_low ,inputs_close), axis=-1)
target = np.concatenate((target_high, target_low,target_close), axis=-1)



# Split the data into training and testing sets
train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs, target, test_size=0.1)

# Print the shapes of the resulting arrays
print('Train inputs shape:', train_inputs.shape)
print('Train targets shape:', train_targets.shape)
print('Test inputs shape:', test_inputs.shape)
print('Test targets shape:', test_targets.shape)



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
model.fit([train_inputs, train_inputs[:, 239:240, :]], train_targets, epochs=300, batch_size=64,
           validation_data=([test_inputs, test_inputs[:, 239:240, :]], test_targets), callbacks=[checkpoint, early_stopping])




# Use the loaded model for prediction

model = load_model('finance.h5')
predictions = model.predict([test_inputs, test_inputs[:, 239:240, :]])
print(predictions.shape)

print(predictions)
