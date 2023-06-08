#include <EloquentTinyML.h>
#include "model.h"
//#include <Arduino_HTS221.h>
#include <Arduino_HS300x.h>
#include <Arduino_LPS22HB.h>
#include <Array.h>

#include <EloquentTinyML.h>
#include <eloquent_tinyml/tensorflow.h>

#define NUMBER_OF_INPUTS 3
#define NUMBER_OF_OUTPUTS 3
#define TENSOR_ARENA_SIZE 8*1024

//Eloquent::TinyML::TfLite<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> ml;
Eloquent::TinyML::TensorFlow::TensorFlow<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> ml;

void setup() {
  Serial.begin(9600);
  Serial.println("Communication started");
  if (!HS300x.begin()) {
    Serial.println("Failed to initialize humidity temperature sensor!");
    while (1);
  }
  if (!BARO.begin()) {
    Serial.println("Failed to initialize pressure sensor!");
    while (1);
  }
  ml.begin(model);
}

void loop() {
  //float temperature = HTS.readTemperature();
  //float humidity = HTS.readHumidity();
  float temperature = HS300x.readTemperature();
  float humidity    = HS300x.readHumidity();
  float pressure = BARO.readPressure() * 10;

  float input[NUMBER_OF_INPUTS] = {temperature, humidity, pressure};
  float output[NUMBER_OF_OUTPUTS] = {0, 0, 0};
  ml.predict(input, output);

  Serial.print("Temperature : ");
  Serial.print(temperature);
  Serial.print(" °C Humidity : ");
  Serial.print(humidity);
  Serial.print(" % Pressure : ");
  Serial.println(pressure);
  Serial.println("The sky is : ");
  
  //float output[5] = {0, 0, 0, 0, 0};
  for (int i = 0; i < 3; i++) {
  Serial.print("Output[");
  Serial.print(i);
  Serial.print("]: ");
  Serial.println(output[i]);
}



  int maxIndex = findMaxIndex(output, 3);
  Serial.println("maxIndex is : ");
  Serial.println(maxIndex);
  //s1 = output;
  //String weatherCondition = abc(maxIndex);
  //Serial.println(weatherCondition);
  if (maxIndex == 0)
    Serial.println("The weather is cloudy!");
  if (maxIndex == 1)
    Serial.println("The weather is clear!");
  if (maxIndex == 2)
    Serial.println("The weather is rainy!");
  delay(1000);
}




/*
String value(float out[5]) {
  Array<int> my_classes = Array<int>(out, NUMBER_OF_OUTPUTS);
  //Array<float, NUMBER_OF_OUTPUTS> my_classes = Array<float, NUMBER_OF_OUTPUTS>(out);

  if (my_classes.getMaxIndex() == 0) {
    return "Clear";
  } else if (my_classes.getMaxIndex() == 1) {
    return "Partly cloudy";
  } else if (my_classes.getMaxIndex() == 2) {
    return "Mostly cloudy";
  } else if (my_classes.getMaxIndex() == 3) {
    return "Overcast";
  } else {
    return "Foggy";
  }
}
*/
int findMaxIndex(float* arr, int size) {
  float maxValue = arr[0];
  int maxIndex = 0;

  for (int i = 1; i < size; i++) {
    if (arr[i] > maxValue) {
      maxValue = arr[i];
      maxIndex = i;
    }
  }

  return maxIndex;
}
