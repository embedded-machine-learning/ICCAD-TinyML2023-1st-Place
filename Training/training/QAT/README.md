# Fast QAT for Power Of Two Rescaler

The goal of the project is to freely define the rescaler of a quantized neural network. 

## What is the rescaler of a Neural Network
The rescaler scales the accumulation register down to the activation size. It is part of an activation function. 
Simple activation functions such as ReLu, ReLu6, and PACT consist of a clipping operation(non linear) and a scaling operation(rescaler)
Rescalers come in different forms, such as full precision multiplication to shifting operations. 


## How to use it
There are examples in ./example , they explain the most important aspects. 

## CITE
If you use this, please cite us. As the paper is not yet publicly released it will be in the proceedings of DSD2023 under the Title: "Fast, Quantization Aware DNN Training for Efficient HW Implementation", Authors: Daniel Schnoell, Matthias Wess, Matthias Bittner, Maximilian Götzinger and Axel Jantsch. </br>
As soon as there is an official link we will update this note.
