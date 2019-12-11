import pandas as pd
# To hide warning "Using ____ backend"
import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.utils.np_utils import to_categorical
import tensorflow as tf
sys.stderr = stderr

# Load the data
dataset = pd.read_csv('7_23c.csv')
# Correlation Matrix for target
corr_matrix = dataset.corr()
print(corr_matrix["A"])

del_col = [5, 6, 7, 15, 19, 21, 22, 23, 24, 26, 27, 28]
# Drop unnecessary columns
dataset.drop(dataset.columns[del_col], axis=1, inplace=True)
print(dataset.head())

val_dataset = pd.read_csv('7_23c.csv')
# Drop unnecessary columns
val_dataset.drop(val_dataset.columns[del_col], axis=1, inplace=True)

# Get Pandas array value (Convert to NumPy array)
train_data = dataset.values
val_data = val_dataset.values

# Use columns 2 to last as Input
train_x = train_data[:,2:]
val_x = val_data[:,2:]

# Use columns 1 as Output/Target (One-Hot Encoding)
train_y = to_categorical( train_data[:,1] )
val_y = to_categorical( val_data[:,1] )

model = Sequential()
model.add(Dense(30, input_dim=15, activation='softmax'))
model.add(Dense(20, activation='softmax'))
model.add(Dense(10, activation='softmax'))
model.add(Dense(5, activation='softmax'))
model.add(Dense(3, activation='softmax'))
opt = Adam(lr = 0.01)
sgd = SGD(lr=0.01, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt, \
                    metrics=['accuracy'])

history = model.fit(train_x, train_y, \
                    validation_data=(val_x, val_y), \
                    epochs=100, batch_size=10, verbose=1)
_,accuracy = model.evaluate(train_x, train_y, \
                verbose=1)
                
predict = model.predict(val_x)
# Visualize Prediction
pd.set_option('display.max_rows', 119)
df = pd.DataFrame(predict)
df.columns = [ 'Strength', 'Agility', 'Intelligent' ]
df.index = val_data[:,0]
print(df)
print(accuracy)

