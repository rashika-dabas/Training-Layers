from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


from trainable_layers import define_dense_model_with_hidden_layers, set_layers_to_trainable

def test_define_dense_model_with_hidden_layers():
    """Test the define_dense_model_with_hidden_layers function."""
    model = define_dense_model_with_hidden_layers(12, ['sigmoid', 'sigmoid'], [51, 21], 'softmax', 11)
    assert len(model.layers) == 3
    assert model.layers[0].input_shape == (None, 12)
    assert model.layers[0].output_shape == (None, 51)
    assert model.layers[1].input_shape == (None, 51)
    assert model.layers[1].output_shape == (None, 21)
    assert model.layers[2].input_shape == (None, 21)
    assert model.layers[2].output_shape == (None, 11)
    assert model.layers[0].trainable == True
    assert model.layers[1].trainable == True
    assert model.layers[2].trainable == True

def test_set_layers_to_trainable():
    """Test the set_layers_to_trainable function."""
    model = define_dense_model_with_hidden_layers(12, ['sigmoid', 'sigmoid'], [51, 21], 'softmax', 11)
    model = set_layers_to_trainable(model, [0, 2])
    assert model.layers[0].trainable == True
    assert model.layers[1].trainable == False
    assert model.layers[2].trainable == True
    model = set_layers_to_trainable(model, [1])
    assert model.layers[0].trainable == False
    assert model.layers[1].trainable == True
    assert model.layers[2].trainable == False
    