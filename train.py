from config import cfg
from data_generator2 import Generator
from model import CNN
import tensorflow as tf


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
save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cfg.model_path,
                                   monitor='val_acc',
                                   verbose=0,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='min',
                                   )

# 用自己写的generator
model.fit(train_generator, epochs=1, verbose=1, callbacks=[save_callback])

# x, y = train_generator[0]
# print(y.shape)
# print(y)












