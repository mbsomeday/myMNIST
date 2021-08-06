import numpy as np
from config import cfg
from tensorflow import keras


'''
    坑：Generator一开始没继承 keras.utils.Sequence，
    报错：AttributeError: 'Generator' object has no attribute 'shape'
'''
class Generator(keras.utils.Sequence):
    def __init__(self, dataset, img_size, channels, batch_size, num_classes, shuffle=True):
        self.dataset = dataset
        self.img_size = img_size
        self.channels = channels
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_samples = self.get_num_samples()
        self.shuffle = shuffle
        self.on_epoch_end()
        
        
    def __len__(self):
        with np.load(r'./mnist.npz') as f:
            x_train, y_train = f['x_train'], f['y_train']
        num = x_train.shape[0]
        if self.dataset == "train":
            return int(np.floor(num*0.8 / self.batch_size))
        elif self.dataset == "val":
            return int(np.floor(num*0.2 / self.batch_size))
    
    
    def on_epoch_end(self):
        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)


    def get_num_samples(self):
        with np.load(r'./mnist.npz') as f:
            x_train, y_train = f['x_train'], f['y_train']
        num = x_train.shape[0]
        if self.dataset == "train":
            return num * 0.8
        elif self.dataset == "val":
            return num * 0.2
    
    
    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]
        indices = indices.astype("int32")
        batch_images, batch_labels = self.get_batch_data(indices)
        return batch_images, batch_labels
        
    
    def get_batch_data(self, indices):
        with np.load(r'./mnist.npz') as f:
            x_train, y_train = f['x_train'], f['y_train']
        x = np.zeros(shape=(self.batch_size, *self.img_size, self.channels), dtype="float32")
        y = np.zeros(shape=(self.batch_size, self.num_classes))
        
        for idx, img_name in enumerate(indices):
            img = x_train[img_name]
            img = np.expand_dims(img, -1)
            img = img / 255.0
            x[idx] = img
            y[idx] = self.one_hot(y_train[img_name])
        return x, y
    
    
    def one_hot(self, label):
        y = np.zeros(shape=(self.num_classes,), dtype="int32")
        y[label] = 1
        return y
        
    


if __name__ == '__main__':
    train_generator = Generator(dataset="train",
                                  img_size=cfg.img_size,
                                  channels=cfg.channels,
                                  batch_size=cfg.batch_size,
                                  num_classes=cfg.num_classes,
                                  shuffle=True)
    val_generator = Generator(dataset="val",
                                  img_size=cfg.img_size,
                                  channels=cfg.channels,
                                  batch_size=cfg.batch_size,
                                  num_classes=cfg.num_classes,
                                  shuffle=True)
    
    x, y = train_generator[0]
    print(y)
    # with np.load(r'./mnist.npz') as f:
    #     x_train, y_train = f['x_train'], f['y_train']
    # print(x_train.shape)
    # cv2.imshow("img", x_train[0])
    # cv2.waitKey(0)















