'''

TODO


-> Add flags
    -> Is_Training
    -> Save_Model

'''

'''

Main File Pseudo Code (Logical Flow) 

-> Initialize are you training or only forward propagation
(Xiao Lei)
-> If training
    -> Initialize neural network accordingly
    -> Load training data (data set pixels and output matrix)
    -> Train the network
    -> Save model on every iteration
(Anish)
-> If not training
    -> Initialize neural network accordingly
    -> Load the model
    -> Load the perception screen
    -> While True: (eventually we can have key press to terminate loop)
        -> Get screenshot pixels
        -> Run neural network
        -> Feed predictions into perception screen
    
'''