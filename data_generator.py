import numpy as np
from tensorflow import keras
from config import cfg
import cv2
import abc


def get_mnist_data(cfg):
    with np.load(cfg.mnist_path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)

def preprocess_data(images, labels, cfg):
    images = images.astype("float32") / 255
    images = np.expand_dims(images, axis=-1)
    
    labels = keras.utils.to_categorical(labels, cfg.num_classes)
    
    return images, labels


class Generator():
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, mnist_path, img_size, channels, batch_size, num_classes, shuffle=True):
        self.mnist_path = mnist_path
        self.batch_size = batch_size
        self.channels = channels
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_samples = self.get_num_samples()
        self.shuffle = shuffle
        self.on_epoch_end()
    
    
    @abc.abstractmethod
    def get_num_samples(self):
        pass
     
     
    def on_epoch_end(self):
        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
        
    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]
        x, y = self.get_batch_data(indices)
        return x, y


    def get_batch_data(self, indices):
        with np.load(cfg.mnist_path) as f:
            x_train, y_train = f['x_train'], f['y_train']
        x = np.zeros((self.batch_size, *self.img_size, self.channels))
        y = np.zeros((self.batch_size, self.num_classes))
        
        for idx, name in enumerate(indices):
            img = self.data_preprocess(x_train[name])
            x[idx] = img
            y[idx] = self.one_hot(y_train[name])
            
        return x, y
    
    
    def data_preprocess(self, images):
        images = np.expand_dims(images, -1)
        return images
    
    
    def one_hot(self, label):
        res = np.zeros((self.num_classes), dtype="int32")
        res[label] = 1
        return res
        
        

class TrainGenerator(Generator):
    def __len__(self):
        with np.load(cfg.mnist_path) as f:
            x_train, y_train = f['x_train'], f['y_train']
            # x_test, y_test = f['x_test'], f['y_test']
            self.num_samples = x_train.shape[0]
        return int(np.floor(self.num_samples / self.batch_size))
        
        
    def get_num_samples(self):
        with np.load(cfg.mnist_path) as f:
            x_train, y_train = f['x_train'], f['y_train']
        return x_train.shape[0]
    
    
if __name__ == '__main__':
    g = TrainGenerator(mnist_path=cfg.mnist_path,
                       img_size=cfg.img_size,
                       batch_size=cfg.batch_size,
                       num_classes=cfg.num_classes,
                       channels=cfg.channels,
                       shuffle=False)
    l = len(g)
    # print(l)
















