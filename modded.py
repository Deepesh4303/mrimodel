import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense,Reshape
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import ModelCheckpoint
import scipy.io as sio

# Prepare the data
num_samples = 112
height, width, channels = 256, 256, 4

# Assuming you have a list of 112 images, each with dimensions 256x256x4
image_list1 = []  # Your list of images
image_list2 =[]

for i in range(1,113):
    data = sio.loadmat(f"data/actual/MR{i}.mat") 
    data = data[f'MR{i}'] # Load the data from the file (replace with your own method)
    image_list1.append(data)
    data2 = sio.loadmat(f"data/maps/ME{i}.mat")
    data2=data2[f'ME{i}']
    image_list2.append(data2)

# Combine the images into kspace_data and exp_map_data
kspace_data = np.stack([image for image in image_list1], axis=0)
exp_map_data = np.stack([image for image in image_list2], axis=0)

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(kspace_data, exp_map_data, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(height, width, channels)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(height*width*channels, activation='linear'))

model.add(Reshape((height, width, channels)))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)
# Fit the model
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), callbacks=[checkpoint])

# Evaluate the model
loss = model.evaluate(x_test, y_test)
print("Test Loss:", loss)
