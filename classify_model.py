# -*- coding: utf-8 -*-
# @Time    : 2020/11/13
# @Author  : AaronJny
# @File    : classify_model.py
# @Desc    :
import tensorflow as tf
from tensorflow.keras import layers

from config import ClassifyConfig


class SiameseNetwork(tf.keras.Model):
    # Input：输入层，用于接收输入数据。
    # Dense：全连接层，将输入和输出之间的所有神经元都相互连接。
    # Conv2D：二维卷积层，用于处理图像和视频数据。
    # MaxPooling2D：二维最大池化层，用于对卷积层的输出进行下采样。
    # Flatten：扁平化层，将多维输入数据展平成一维。
    # Dropout：随机失活层，用于防止过拟合。
    # BatchNormalization：批量归一化层，用于加速神经网络的训练和提高模型的精度。
    # Activation：激活函数层，用于为神经网络添加非线性变换。
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.net_stage_1 = tf.keras.Sequential([
            layers.Input(shape=(*ClassifyConfig.IMAGE_SIZE, ClassifyConfig.IMAGE_CHANNELS)),
            layers.Conv2D(6, (3, 3), padding='same'),  # padding: "valid", "causal" 或 "same" 之一 (大小写敏感) "valid" 不填充, "same" 表示填充输入以使输出具有与原始输入相同的长度。
            layers.MaxPooling2D((2, 2), 2),
            layers.Dropout(ClassifyConfig.DROPOUT_RATE),
            layers.ReLU(),
            layers.Conv2D(16, (5, 5)),
            layers.MaxPooling2D((2, 2), 2),
            layers.Dropout(ClassifyConfig.DROPOUT_RATE),
            layers.ReLU()
        ])
        self.net_stage_2 = tf.keras.Sequential([
            layers.Conv2D(6, (3, 3)),
            layers.MaxPooling2D((2, 2), 2),
            layers.Dropout(ClassifyConfig.DROPOUT_RATE),
            layers.ReLU(),
            layers.Flatten(),
            layers.Dense(84),
            # layers.Dropout(ClassifyConfig.DROPOUT_RATE),
            layers.ReLU(),
            layers.Dense(1, activation='sigmoid')
        ])

    @tf.function
    def call(self, inputs, training=None, mask=None):
        outs = []
        for x in inputs:
            out = self.net_stage_1(x)
            outs.append(out)
        out = tf.concat(outs, axis=-1)
        out = self.net_stage_2(out)
        return out


def load_classify_model():
    model = SiameseNetwork()
    model.build(
        [(None, *ClassifyConfig.IMAGE_SIZE, ClassifyConfig.IMAGE_CHANNELS),
         (None, *ClassifyConfig.IMAGE_SIZE, ClassifyConfig.IMAGE_CHANNELS)])
    model.load_weights(ClassifyConfig.MODEL_PATH)
    return model
