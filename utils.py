import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint


class myModelCheckpoint(ModelCheckpoint):
    
    def __init__(self, filtpath, alternative_model, **kwargs):
        self.alternative_model = alternative_model
        super().__init__(filtpath, **kwargs)
        
    def on_epoch_end(self, epoch, logs=None):
        print("当前为epoch end")
        model_before = self.model
        self.model = self.alternative_model
        super().on_epoch_end(epoch, logs)
        self.model = model_before
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

























