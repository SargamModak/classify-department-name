This repository contains my trained model.  
If you run _run.py_ file it will run the model through test dataset (created randomly and not used in training) and 
prints the accuracy score on the test set.  
To train the model from beginning just delete/rename _model.h5_ in _util\_files_ folder.

Description of different files:
* _create\_dataset.py_: creates training, validation and testing data randomly.
* _create\_model.py_: creates and returns the model.
* _datagen.py_: reads the data and returns the generator function for training and validation.
* _run.py_: main file to train and test or just testing the model.
* _test.py_: to predict and validate the model on test set.