import numpy as np
import os
import random
from matplotlib import pyplot as plt
from PIL import Image
import deeplearning as dl
from scipy import interpolate

class NN1D():

    def __init__(self, input_dims, batch_size, dropout=0):

        self.input_dims = input_dims
        self.batch_size = batch_size
        self.dropout = dropout

        self.layer_outputs=[]
        self.lastgrad = []
        
        # Initial feature extraction, double input size
        self.layer0 = dl.nnlayer(input_size=1024, output_size=2048, batch_size=self.batch_size)

        # 2048x1 input -> 2048x512 weight -> 512x1 features
        self.layer1 = dl.nnlayer(input_size=2048, output_size=512, batch_size=self.batch_size)
        
        # 512x1 input -> 512x64 weight -> 64x1 features
        self.layer2 = dl.nnlayer(input_size=512, output_size=64, batch_size=self.batch_size)

        # 64x1 input -> 64x1 weight -> 1x1 features
        self.layer3 = dl.nnlayer(input_size=64, output_size=1, batch_size=self.batch_size)
        
        self.activation = dl.activation('relu', self.batch_size)

        self.sigmoid = dl.activation('sigmoid', self.batch_size)

        self.drop = dl.dropout(self.dropout)


    def forward(self, x, state):

        z1 = self.layer0.forward(x)
        a1 = self.activation.forward(z1)
        a1 = self.drop.forward(a1,state)
        
        z2 = self.layer1.forward(a1)
        a2 = self.activation.forward(z2)
        a2 = self.drop.forward(a2,state)

        z3 = self.layer2.forward(a2)
        a3 = self.activation.forward(z3)
        a3 = self.drop.forward(a3,state)
        
        z4 = self.layer3.forward(a3)

        self.y_hat = self.sigmoid.forward(z4)

        
        self.layer_outputs= [(0,x), (z1,a1), (z2,a2), (z3,a3), (z4,self.y_hat)]

        self.y_hat = [yhat[0].flatten() for yhat in self.y_hat]
        self.y_hat = np.array(self.y_hat).T

        return self.y_hat.T

    def backward(self,y_true,lr):
        
        self.lastgrad = []
        
        for l in range(4,0,-1):

            if l == 4:

                d = (self.y_hat-y_true).T
                a_l1 = self.layer_outputs[l-1][1]

                dL_dW = [d[batch] * a_l1[batch].T for batch in range(self.batch_size)]
                dL_dW = np.mean(dL_dW, axis=0)

            elif l == 3:

                dL_da = [d[batch] * self.layer3.w for batch in range(self.batch_size)]
                lz = self.layer_outputs[l][0]
                a_l1 = self.layer_outputs[l-1][1]

                da_dz = self.sigmoid.forward(lz)

                d = [dL_da[batch] * da_dz[batch].T for batch in range(self.batch_size)]

                dL_dW = [np.matmul(d[batch].T, a_l1[batch].T) for batch in range(self.batch_size)]

                dL_dW = np.mean(dL_dW, axis=0)

            elif l == 2:

                dL_da = [np.matmul(d[batch],self.layer2.w) for batch in range(self.batch_size)]
                lz = self.layer_outputs[l][0]
                a_l1 = self.layer_outputs[l-1][1]

                da_dz = self.sigmoid.forward(lz)

                d = [dL_da[batch] * da_dz[batch].T for batch in range(self.batch_size)]

                dL_dW = [np.matmul(d[batch].T, a_l1[batch].T) for batch in range(self.batch_size)]

                dL_dW = np.mean(dL_dW, axis=0)

            elif l == 1:

                dL_da = [np.matmul(d[batch],self.layer1.w) for batch in range(self.batch_size)]
                lz = self.layer_outputs[l][0]
                a_l1 = self.layer_outputs[l-1][1]

                da_dz = self.sigmoid.forward(lz)

                d = [dL_da[batch] * da_dz[batch].T for batch in range(self.batch_size)]

                dL_dW = [np.matmul(d[batch].T, a_l1[batch].T) for batch in range(self.batch_size)]

                dL_dW = np.mean(dL_dW, axis=0)

            else:
                return

            dL_dW = dL_dW/abs(dL_dW).max()
            d_l1 = np.mean(d, axis=0)
            d_l1 = d_l1/abs(d_l1).max()
            self.step(lr, (dL_dW, d_l1), l)
            self.lastgrad.append((dL_dW, d_l1))

        return

    def step(self,lr,grad,i, moment=0.9):

        if len(self.lastgrad)==4:
            grad[0] = moment*self.lastgrad[i][0] + (1-moment)*grad[0]
            grad[1] = moment*self.lastgrad[i][1] + (1-moment)*grad[1]
            pass

        if i == 4:
            self.layer3.w -= lr*grad[0]
            self.layer3.b -= lr*grad[1]
        elif i == 3:
            self.layer2.w -= lr*grad[0]
            self.layer2.b -= lr*grad[1].T
        elif i == 2: 
            self.layer1.w -= lr*grad[0]
            self.layer1.b -= lr*grad[1].T
        elif i == 1:
            self.layer0.w -= lr*grad[0]
            self.layer0.b -= lr*grad[1].T
        else:
            return
        
def main():
    
    x_train, train_labels = get_data('./training', 'png', ['neg','pos'], [0,1])
    x_val , val_labels = get_data('./validation', 'png', ['neg','pos'], [0,1])

    train_params = { 
                    'lr':1e-3,
                    'epochs':35,
                    'batch_size':32,
                    'dropout':0.1
                    }

    im_size = 1024

    _model = NN1D(1, train_params['batch_size'], train_params['dropout'])
    data={'training':x_train, 'train_Labels':train_labels, 'validation':x_val , 'val_labels':val_labels }
    training(data,train_params,_model)

def training(data,train_params,model):

    x_train = data['training']
    y_train = data['train_Labels']
    x_val =  data['validation']
    y_val = np.array(data['val_labels'])

    batch_size = train_params['batch_size']

    no_batches = (x_train.shape[0])//batch_size

    train_loss_hist = []
    train_accuracy_hist = []
    val_loss_hist = [1]
    val_accuracy_hist = [0]

    for t in range(train_params['epochs']):

        # lr dropfactor

        if t == 30:
            train_params['lr'] = train_params['lr']*0.1

        for k in range(no_batches):

            x_batch = x_train[k*batch_size:(k+1)*batch_size,:]
            y_batch = np.array(y_train[k*batch_size:(k+1)*batch_size])

            y_pred = model.forward(x_batch, 'training')

            loss = dl.binary_crossentropy(y_batch, y_pred)
            acc = dl.binary_acc(y_batch, y_pred)

            # Backward pass
            # Update parameters
            model.backward(y_batch,train_params['lr'])
        
            train_loss_hist.append(loss)
            train_accuracy_hist.append(acc)
            # print(acc, loss)

        y_pred_val = model.forward(x_val,'evaluate')

        val_loss = dl.binary_crossentropy(y_val, y_pred_val)
        val_acc = dl.binary_acc(y_val, y_pred_val)

        val_loss_hist.append(val_loss)
        val_accuracy_hist.append(val_acc)


    x = np.arange(0,(train_params['epochs']+1)*no_batches,no_batches)
    f_loss = interpolate.interp1d(x,val_loss_hist)
    f_acc = interpolate.interp1d(x,val_accuracy_hist)

    new_x = np.arange(0,(train_params['epochs'])*no_batches,1)
    val_acc_new= f_acc(new_x)
    val_loss_new= f_loss(new_x)

    print("best score:", max(val_acc_new))

    plt.subplot(2,1,1)
    plt.plot(train_accuracy_hist)
    plt.plot(val_acc_new)
    plt.ylabel('Accuracy')
    plt.xlabel('Batch No.')
    plt.legend(['training','validation'], loc='upper left')
    plt.subplot(2,1,2)
    plt.plot(train_loss_hist)
    plt.plot(val_loss_new)
    plt.ylabel('Loss')
    plt.xlabel('Batch No.')
    plt.legend(['training','validation'], loc='upper left')
    plt.show()

def get_data(file_path,obj_type,class_names,class_num):
    
    # sorting a label list, and total objects size
    
    file_names =os.listdir(file_path)
    random.shuffle(file_names)
    labels = []
    obj_size = len(file_names)  
    file_lst = [file_names[i].split('_')[0] for i in range(len(file_names))]


    labels = [class_num[i_name]  for name in file_lst for i_name in range(len(class_names))  if name == class_names[i_name]]
    
    # matrix of row vector of pictures
    if obj_type == 'png':
        im = [np.array(Image.open(file_path + "/" + name)) for name in file_names]
        lines = [np.reshape(line,[im[0].size,1]) for line in im]
        lines=np.array(lines)/255
         
    return lines, labels

if __name__ == "__main__":

    np.random.seed(7)
    main()
