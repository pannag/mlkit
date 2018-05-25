# mlkit
Sample code and apps for the ML Kit (https://firebase.google.com/docs/ml-kit/). 

MLKitCusomt is a simple Android app that runs a tflite model. The tflite model takes a 64-dim input and returns the sum of all the inputs. This is like a
 logistic regression model without the softmax layer and where all the weights are 1.0 and bias is
 0.

## Run the MLKitCustom app
You can run the Android app and test the model out. You can enter a number that will be replicated 64 times and passed as
input to the model. The expected output is 64 * input number. If no input is given, then 64
random numbers are used as input to the model.

## Create your own new model
If you want, you can create your own TFLite model and update the Android app to use that. 

First you will have to copy and update the the *sum_inputs_tflite.py* to create your new model file. 
Let's say the new file is called *new_model_tflite.py*. Then:
    
    cd python
    python new_model_tflite.py
   
The script currently saves the *frozen_graph.pbtxt* file as well. That can be printed out to see if the graph makes sense.

Then you have to drag the model into the *assets* folder of the Android app, and then update the inputs and outputs dimensions
in the *MainActivity.java*.

## Use existing graphdef model 
If you had a frozen graphdef already from somewhere else, you can use that as well. You will have to comment out
the model definition parts from the python script (or use commandline Toco directly). Before you do that, **it is 
useful to look inside the graph file and understand your inputs and outputs**.
You can either load it up in *Tensorboard* or print out the nodes as follows:
    
It is useful to look into the graphdef as well.

    for node in frozen_graph_def.node:
      print(node.name)
 
You can also convert the frozen graphdef .pb to text readable .pbtxt as follows.
 
    import tensorflow as tf
    path_to_pb = '...'
    output_file = '...'
    g = tf.GraphDef()
    with open(path_to_pb, 'r') as f:
      g.ParseFromString(str(f.read()))
    with open(output_file, 'w') as f:
      f.write(str(g))
  
## Use existing tflite model 
If you already have a tflite model, then you can skip the python step and directly go to the Android app. Before you do that, it is 
**useful to look inside the tflite file and understand your inputs and outputs**. To do that, we can use *flatc* tool and convert
the model to *.json* file and read that. 

First clone the flatbuffers repo and build flatc.
    
    git clone https://github.com/google/flatbuffers.git
   
Then you have to have the tensorflow schema.fbs stored locally. Either checkout the tensorflow github or download
that [one file[(https://github.com/tensorflow/tensorflow/blob/18003982ff9c809ab8e9b76dd4c9b9ebc795f4b8/tensorflow/contrib/lite/schema/schema.fbs). 
Then you can run *flatc* to generate the json file from then input tflite model.
    
    flatc -t schema.fbs -- input_model.tflite
    
 This will create a *input_model.json* file that can be easily read. 

