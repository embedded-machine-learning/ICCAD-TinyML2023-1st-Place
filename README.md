# Training
### Environment setup
Go into the `Training` directory.  
Create an python environment and install the requirements file.
Copy the `tinyml_contest_data_training` folder into the directory.

### Training
simply run:  
`python training/train.py`  
(Please note that we are still in the `Training` directory, a subdirectory is called `training` )  
A new folder `saved_models` is created where the model files are stored.  
You need to select your optimal model.

We are using our QAT repository https://github.com/embedded-machine-learning/FastQATforPOTRescaler to get a power of two rescaler. It should be published as part of DSD23, however as this is not available yet https://jantsch.se/AxelJantsch/papers/2023/DanielSchnoell-DSD.pdf .

### Weight export
Open the notebook `netron_tens/nunpy_to_c_QAT.ipynb`  
You need to select the model in the first cell, then simply run the whole notebook and a new file `wights_Quant.hpp` is created. This file needs to be copied into the c Project.

# Compile C Project+
Inside `CProject` open `TESTMODEL` in MDK5  
Inside MDK5 project the following options need to be set:
* **Arm Compiler Version 6.18** (essential)
* c compiler flags `-Omin`
* Activate Link Time Optimization
* Use Oz
* Language C: gnu11
* Language C++: gnu++17 (community)

Please double check these settings with the following images:

![Settings A](./images/photo_5836991556517740854_x.jpg)
![Settings B](./images/12684041-9de3-4229-b567-f84e887973c6.jpeg)


# Updates since Contest
The original solution is in `submission.zip` \
The end of the contest got a bit stressful, so bugs where introduced. 
They modified how the compiler interpreted the code, which lead to a suboptimal translation. 
The update removed these inaccuracies, while of course not modifying the results of the neural network. \
The biggest difference is the usage of the template Layers rather than the classical C type, type dynamic function, which improves readability. 
By using constant weights, as originally intended and easily interpretable code the compiler can optimize better.  

<pre>
Program Size: 
Code=       9458B  →  9518B    (+60)    [Data-type change, two template instances rather than one function]
RO-data=    3446B  →  3530B    (+84)    [Moved data from RW to RO, and changed one datatype
RW-data=     236B  →    20B   (-216)       from int32 to int8.]
ZI-data=   10492B  → 10492B     (±0)
Latency=    1.52ms →  1.48ms (-0.04)    [no void pointers, fully known data-types, only
                                           compile-time dynamic, but runtime static.]    
</pre>
The code size change is actually very interesting. About 36 bytes come from the change of the data-type form the right shift (with the ugly functions). It now needs to load a quarter words, which does not natively exist (to the best of our knowledge). But the change of type reduces the size of the data by 66 bytes (3*22B). (This code overhead is also true for 16 bit integer even though a native half world load exists.)