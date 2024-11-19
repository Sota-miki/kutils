from . import generic, tensor_ops, image_utils
from . import generators, model_helper, applications

from tensorflow.keras import backend as K
import tensorflow as tf

if K.backend() == 'tensorflow': 
    # TensorFlow 2.x ではチャンネルの順序はこのように設定します
    tf.keras.backend.set_image_data_format("channels_last")  # "tf" に相当する設定


# remove tensorflow warning
import logging
class WarningFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        tf_warning = 'retry' in msg
        return not tf_warning           
logger = logging.getLogger('tensorflow')
logger.addFilter(WarningFilter())

# if too many warnings from scikit-image 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

print('loaded kutils')