'''
模型reuse：

https://blog.csdn.net/laplacebh/article/details/107088656

model = MyModel()  # 实例化
model(tf.ones(shape=INPUT_SHAPE))  # 随便用个输入跑一下，初始化模型
model.load_weights('model_name.h5')  # 加载权重

'''