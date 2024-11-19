import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import os


# Keras configuration directives

def SetActiveGPU(number=0):
    """
    Set visibility of GPUs to the Tensorflow engine.

    :param number: scalar or list of GPU indices
                   e.g. 0 for the 1st GPU, or [0,2] for the 1st and 3rd GPU
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if not isinstance(number,list):
        number=[number]
    os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join(map(str,number))
    print('Visible GPU(s):', os.environ["CUDA_VISIBLE_DEVICES"])

def GPUMemoryCap(fraction=1):
    """
    Limit the amount of GPU memory that can be used by an active kernel.

    :param fraction: in [0, 1], 1 = the entire available GPU memory.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=fraction * tf.config.experimental.get_memory_info(gpu)['total_memory']
                    )]
                )
            print(f"Set GPU memory fraction to {fraction}.")
        except RuntimeError as e:
            print(e)


# Metrics and losses
    
def plcc_tf(x, y):
    """PLCC metric"""
    xc = x - tf.reduce_mean(x)
    yc = y - tf.reduce_mean(y)
    return tf.reduce_mean(xc * yc) / (tf.math.reduce_std(x) * tf.math.reduce_std(y) + K.epsilon())

def earth_mover_loss(y_true, y_pred):
    """
    Earth Mover's Distance loss.

    Reproduced from https://github.com/titu1994/neural-image-assessment/blob/master/train_inception_resnet.py
    """
    cdf_ytrue = tf.cumsum(y_true, axis=-1)
    cdf_ypred = tf.cumsum(y_pred, axis=-1)
    samplewise_emd = tf.sqrt(tf.reduce_mean(tf.square(tf.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return tf.reduce_mean(samplewise_emd)

def make_loss(loss, **params_defa):
    def custom_loss(*args, **kwargs):
        kwargs.update(params_defa)
        return loss(*args, **kwargs)
    return custom_loss

