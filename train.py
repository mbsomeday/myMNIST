from config import cfg
from data_generator2 import Generator, getDataInCache
from model import CNN
import tensorflow as tf
from utils import myModelCheckpoint


train_generator = Generator(dataset="train",
                            img_size=cfg.img_size,
                            channels=cfg.channels,
                            batch_size=cfg.batch_size,
                            num_classes=cfg.num_classes,
                            shuffle=True)

model = CNN(cfg)
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# callback函数
# save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cfg.model_path,
#                                    monitor='val_acc',
#                                    verbose=0,
#                                    save_best_only=True,
#                                    save_weights_only=False,
#                                    mode='min',
#                                    )


# ModelCheckpoint的子类
ckptCallback = myModelCheckpoint(cfg.model_path, model)

# 用自己写的generator
# model.fit(train_generator, epochs=1, verbose=1, callbacks=[save_callback])


# 将数据全部存到缓存
(x_train, y_train), (x_test, y_test) = getDataInCache(cfg.mnist_path)
model.fit(x_train, y_train, batch_size=cfg.batch_size, epochs=2, validation_split=0.1, callbacks=[ckptCallback])


# x, y = train_generator[0]
# print(y.shape)
# print(y)












