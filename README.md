# NumpyNet
Pytorch Style NN from scratch with only Numpy

This Neural network is capable of handling a 1D (1024x1) input vector that represents an MRI Image, and to predict a presence of lesions in the image.
The Neural network utilizes fully connected layers. 
The use of backpropagation gradient descent, batches of training data and dropouts from the training set.

Two main Classes in this code: NN1D and nnlayer. 

#NN1D 
constract the NN model by utilizing the 'nnlayer' class
backward pass function - backpropegation
step function - update gradients
       
#nnlayer  
initilize layers
forward pass function 
loss function
accuracy function 



CNN characteristics:

        input layer = (input_size=1024, output_size=2048, batch_size)
        activation = 'reLU'
        droplayer = (p=dropout)

        # 2048x1 input -> 2048x512 weight -> 512x1 features
        layer1 = (input_size=2048, output_size=512, batch size)
        activation = 'reLU'
        droplayer = (p=dropout)
        
        # 512x1 input -> 512x64 weight -> 64x1 features
        layer2 =(input_size=512, output_size=64, batch size)
        activation = 'reLU'
        droplayer = (p=dropout)

        # 64x1 input -> 64x1 weight -> 1x1 features
        output layer = dl.nnlayer(input_size=64, output_size=1, batch size)
        activation = 'Sigmoid'
        
        #loss function
        binary_crossentropy = -1 * np.sum((np.dot(y_batch,np.log(epsilon + y_pred)) + np.dot((1-y_batch),np.log(epsilon + 1-y_pred))))/len(y_batch)
        
        #optimaizer
        apply basic backpropegation on the model by backwards function
        
        
# example for train parameters used in the CNN 
train_params = { 
                    'lr':1e-3,
                    'epochs':35,
                    'batch_size':32,
                    'dropout':0.1
                    }

im_size = 1024
    

# Tal Grutman & Yariv Edry 
