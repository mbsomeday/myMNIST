import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint


class myModelCheckpoint(ModelCheckpoint):
    
    def __init__(self, filtpath, **kwargs):
        super().__init__(filtpath, **kwargs)
        
    def on_epoch_end(self, epoch, logs=None):
        print("当前为epoch end")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

























