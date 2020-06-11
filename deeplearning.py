import numpy as np

class nnlayer():
    
    def __init__(self, input_size, output_size, batch_size):

        self.input_size = input_size
        self.output_size = output_size 
        self.batch_size = batch_size
        self.w = self.init_weights()
        self.b = self.init_bias()
    
    def forward(self, x):

        wx = np.matmul(self.w, x)+self.b

        return wx


    def init_weights(self):
        weights = -1+2*np.random.rand(self.output_size,self.input_size)
        return weights

    def init_bias(self):
        bias = -1+2*np.random.rand(self.output_size,1)
        return bias


class activation():

    def __init__(self, type, batch_size):
    
        self.type=type  
        self.batch_size=batch_size

    def forward(self,x):

        if self.type=='relu':

            x = x/x.max()
            f_x= np.maximum(0,x)

        elif self.type=='sigmoid':
            
            f_x= 1/(1+np.exp((-1)*x))
        
        return f_x

class dropout():

    def __init__(self, dropout):

        self.dropout = dropout

    def forward(self, x, state):
        if self.dropout == 0 or state=="evaluate":
            return x
        else:
            return x*np.random.choice([0, 1], size=x.shape, p=[self.dropout, 1-self.dropout])

def loss(y_batch, y_pred):
    f_x= np.sum(np.power(y_pred-y_batch,2)/2)
    return f_x

def binary_crossentropy(y_batch, y_pred):
    epsilon = 1e-7
    f_x = -1 * np.sum((np.dot(y_batch,np.log(epsilon + y_pred)) + np.dot((1-y_batch),np.log(epsilon + 1-y_pred))))/len(y_batch)
    return f_x

def binary_acc(y_batch, y_pred):
    y_pred = np.round(y_pred)
    acc= np.equal(y_batch,y_pred.T)  
    len_batch = len(y_batch)
    acc=(np.sum(acc))/len_batch
    return acc

    

    






if __name__ == "__main__":
    import model
    model.main()








    
        
    




        