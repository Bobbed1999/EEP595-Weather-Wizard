# %%
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import subprocess


def load_csv(file_name):
  # Input: csv_file_name
  # Output: pandas data frame
    df = pd.read_csv(file_name, sep = ',')
    return df

humidity= load_csv('humidity.csv')
pressure= load_csv('pressure.csv')
temperature= load_csv('temperature.csv')
label= load_csv('weather_description.csv')

humidity= humidity.drop(humidity.index[0])
pressure= pressure.drop(pressure.index[0])
temperature= temperature.drop(temperature.index[0])

label= label.drop(label.index[0])


all_col= ['Vancouver', 'Portland', 'San Francisco','Seattle', 'Los Angeles', 'San Diego', 'Las Vegas', 'Phoenix', 'Albuquerque', 'Philadelphia', 'New York', 'Montreal', 'Boston', 'Beersheba',	'Tel Aviv District', 'Eilat', 'Haifa', 'Nahariyya',	'Jerusalem']
temperature[all_col]= temperature[all_col]-273.15

t1= pd.merge(left= pressure, right= temperature, on= 'datetime', suffixes=("_pressure", "_tempreture"))

t2= pd.merge(left= t1, right=humidity, on= 'datetime')

all_col= ['Vancouver', 'Portland', 'San Francisco','Seattle', 'Los Angeles', 'San Diego', 'Las Vegas', 'Phoenix', 'Albuquerque', 'Philadelphia', 'New York', 'Montreal', 'Boston', 'Beersheba',	'Tel Aviv District', 'Eilat', 'Haifa', 'Nahariyya',	'Jerusalem']
all_col_hum= []
for col in all_col: 
    all_col_hum.append(col+ "_humidity")

for i in all_col:
    t2.rename(columns={i: all_col_hum[all_col.index(i)]}, inplace=True)

t2= pd.merge(left= t2, right=label, on= 'datetime')


seattle= t2[["Seattle_pressure", "Seattle_humidity", "Seattle_tempreture", "Seattle"]]
seattle = seattle.rename(columns={'Seattle_pressure': 'pressure', 'Seattle_tempreture':'tempreture', 'Seattle_humidity':'humidity','Seattle':'label'},)

portland= t2[["Portland_pressure", "Portland_humidity", "Portland_tempreture", "Portland"]]
portland = portland.rename(columns={'Portland_pressure': 'pressure', 'Portland_tempreture':'tempreture', 'Portland_humidity':'humidity','Portland':'label'},)

San_Diego= t2[["San Diego_pressure", "San Diego_humidity", "San Diego_tempreture", "San Diego"]]
San_Diego = San_Diego.rename(columns={'San Diego_pressure': 'pressure', 'San Diego_tempreture':'tempreture', 'San Diego_humidity':'humidity','San Diego':'label'},)

seattle = pd.concat([seattle, portland], ignore_index= True )
seattle = pd.concat([seattle, San_Diego], ignore_index= True )

# test 
new_df = seattle
new_df = seattle[['tempreture', 'humidity', 'pressure','label']]

new_df = new_df.dropna() #remove empty rows

#Convert each category into an integer
for ind in new_df.index:
  if new_df["label"][ind]=='overcast clouds':
    new_df["label"][ind] = 0
  elif new_df["label"][ind]=='sky is clear':
    new_df["label"][ind] = 1
  elif new_df["label"][ind]=='light rain':
    new_df["label"][ind] = 2
  #elif new_df["Seattle"][ind]=='Drizzle':
  #  new_df["Seattle"][ind] = 2
  else:
    new_df = new_df.drop([ind]) #we don't consider other classes so we drop it

#Cast this column to int
new_df["label"] = new_df["label"].astype(int)


#Parameters :
NB_classes = 3 #number of outputs
NB_neurones = 30 #main number of neurones
NB_features = 3 #number of inputs
activation_func = tf.keras.activations.relu #activation function used

#Fully connected neural network
model = tf.keras.Sequential([
                             tf.keras.layers.Dense(9,activation=activation_func,input_shape=(NB_features,)),
                             tf.keras.layers.Dense(27,activation=activation_func),
                             tf.keras.layers.Dense(NB_classes,activation=tf.keras.activations.softmax)
])

model.compile(optimizer="adam",loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy']) #compile the model
# model.build()
model.summary() #to see the paramter of our model


# new_df["label"].value_counts()

label_to_drop = new_df[new_df['label'] == 1].index
drop_samples = np.random.choice(label_to_drop, 20000, replace=False)
df_dropped = new_df.drop(drop_samples)

# df_dropped["label"].value_counts()

from keras.utils import to_categorical

labels = to_categorical(df_dropped.pop('label')) #Create classes from the labels

import numpy as np #import numpy library, used for arithmetic

features = np.array(df_dropped) #convert our dataframe into ndarray, only array type that neural network takes as input


from sklearn.model_selection import train_test_split

#Split the dataset into training set 85% and test set 15%
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.20,shuffle=True) 

model_history = model.fit(x=train_features,
          y=train_labels,
          epochs=50,
          validation_data=(test_features,test_labels),
          verbose=1,
          shuffle=True) #Train our model


metrics = model_history.history

import matplotlib.pyplot as plt

plt.plot(model_history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()

plt.plot(model_history.epoch, metrics['accuracy'], metrics['val_accuracy'])
plt.legend(['accuracy', 'val_accuracy'])
plt.show()

performance=model.evaluate(test_features,test_labels, batch_size=32, verbose=1, steps=None, )[1] * 100
print('Final accuracy : ', round(performance), '%')

model.save("model.h5")


model_size = os.stat('model.h5')
print("Size of model before quantizing: {} bytes".format(model_size.st_size))

test_labels = np.argmax(test_labels,axis=1)
print(test_labels)


# Passing the baseline Keras model to the TF Lite Converter.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Using  the Dynamic Range Quantization.
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Converting the model
tflite_quant_model = converter.convert()
# Saving the model.
with open('dynamic_quant_model.tflite', 'wb') as f:
    f.write(tflite_quant_model)

#Function for evaluating TF Lite Model over Test Images
def evaluate(interpreter):
    prediction= []
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    input_format = interpreter.get_output_details()[0]['dtype']
    
    for i, test_image in enumerate(test_features):
        if i % 100 == 0:
            print('Evaluated on {n} results so far.'.format(n=i))
        test_image = np.expand_dims(test_image, axis=0).astype(input_format)
        interpreter.set_tensor(input_index, test_image)

        # Run inference.
        interpreter.invoke()
        output = interpreter.tensor(output_index)
        predicted_label = np.argmax(output()[0])
        prediction.append(predicted_label)

    print('\n')
    # Comparing prediction results with ground truth labels to calculate accuracy.
    prediction = np.array(prediction)
    print(prediction)
    print(test_labels)
    #accuracy = (prediction == test_labels)
    accuracy = (prediction == test_labels).mean()
    return accuracy

# %%
# Passing the Dynamic Range Quantized TF Lite model to the Interpreter.
interpreter = tf.lite.Interpreter('dynamic_quant_model.tflite') 
# Allocating tensors.
interpreter.allocate_tensors()
# Evaluating the model on the test images.
test_accuracy = evaluate(interpreter)
print('Dynamically Quantized TFLite Model Test Accuracy:', test_accuracy*100)

# C++ file


# %%
MODEL_TFLITE = 'dynamic_quant_model.tflite'
MODEL_TFLITE_MICRO = 'model.cpp'

# !xxd -i {MODEL_TFLITE} > {MODEL_TFLITE_MICRO}
# subprocess.run(['xxd -i' ,MODEL_TFLITE, '>' ,MODEL_TFLITE_MICRO])
os.system("xxd -i {0} > {1}".format(MODEL_TFLITE, MODEL_TFLITE_MICRO))

REPLACE_TEXT = MODEL_TFLITE.replace('/','_').replace('.', '_')

# !sed -i 's/{REPLACE_TEXT}/g_model/g' {MODEL_TFLITE_MICRO}
# subprocess.run(['sed -i','s/{REPLACE_TEXT}/g_model/g', MODEL_TFLITE_MICRO])
os.system("sed -i 's/{0}/g_model/g' {1}".format(REPLACE_TEXT, MODEL_TFLITE_MICRO))
