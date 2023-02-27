# A Computational Modelling Approach for Working Memory Capacity using Convolutional Neural Networks

This repository contains the code associated with my bachelor thesis on computational modelling of working memory. It expands upons Sanjay Manohar's implementation which can be found [here](http://www.smanohar.com/wp/wm/download.html).
## Usage
Detailed instructions can be found in the thesis.
* [optimization_colour_location.py](./optimization_colour_location.py): optimization for choosing the most relevant neurons from the vgg16. Change `images_path` to the desired test set path before running. Colour optimization is default.
* [optimization_mooney.py](./optimization_mooney.py): optimization for choosing the most relevant neurons from the vgg16 for detecting mooney faces. Needs to be run before running the accuracy test for mooney face / no mooney face.
* [normalization.py](./normalization.py): calculates normalization array for the output of the vgg16.
* [single_trial.py](./single_trial.py): single test run
* [accuracy_colour1.py](./accuracy_colour1.py): Accuracy of retrieval for different colour but same location and same shape is tested. Set sizes of 1,2,3,4 are tested.
* [accuracy_colour2.py](./accuracy_colour2.py):  Accuracy of retrieval for different colour but same location and same shape is tested. Set sizes of 1,2,3,4 are tested.
* [accuracy_colour3.py](./accuracy_colour3.py):  Accuracy of retrieval for different colour but same location and same shape is tested. Set sizes of 1,2,3,4 are tested.
* [accuracy_complexity.py](./accuracy_complexity.py):  Accuracy of retrieval for different location but same colour and same shape is tested. Set sizes of 1,2,3,4 are tested.
* [accuracy_location.py](./accuracy_location.py):  Accuracy of retrieval for mooney faces and no mooney faces is tested. Set sizes of 1,2,3,4 are tested.
* [accuracy_mooney.py](./accuracy_mooney.py):  Accuracy of retrieval for low complexity and high complexity is tested. Set sizes of 1,2,3,4 are tested.
