import torch
import torch.nn as nn
from torch_geometric.nn import EdgeConv, global_mean_pool
from torch_cluster import knn_graph
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import torch.optim as optim

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MAB(tf.keras.layers.Layer):
    def __init__(self, num_heads, hidden_units, mlp_hidden_units=128, dropout_rate=0.1, **kwargs):
        super(MAB, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_units = hidden_units
        self.mlp_hidden_units = mlp_hidden_units
    
    def build(self, input_shape):
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, 
                                                            key_dim=self.hidden_units//self.num_heads)
        self.feedforward = tf.keras.Sequential([
            layers.Dense(units=self.mlp_hidden_units, activation="relu"),
            layers.Dense(units=input_shape[-1])
        ])
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        super(MAB, self).build(input_shape)
    
    def call(self, inputs, mask=None):
        attention_output = self.attention(inputs, inputs, attention_mask=mask)[0]
        attention_output = self.layer_norm1(inputs + attention_output)
        feedforward_output = self.feedforward(attention_output)
        block_output = self.layer_norm2(attention_output + feedforward_output)
        return block_output

class TransformerNet(tf.keras.layers.Layer):

    def __init__(self, num_heads, hidden_units, mlp_hidden_units=128, dropout_rate=0.1, **kwargs):
        super(TransformerNet, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_units = hidden_units
        self.mlp_hidden_units = mlp_hidden_units
        self.dropout_rate = dropout_rate

    def __call__(self, inputs, mask=None):
        
        out = MAB(num_heads=self.num_heads, hidden_units=self.hidden_units, mlp_hidden_units=self.mlp_hidden_units, dropout_rate=self.dropout_rate)(inputs)
        out = MAB(num_heads=self.num_heads, hidden_units=self.hidden_units, mlp_hidden_units=self.mlp_hidden_units, dropout_rate=self.dropout_rate)(out)
        out = MAB(num_heads=self.num_heads, hidden_units=self.hidden_units, mlp_hidden_units=self.mlp_hidden_units, dropout_rate=self.dropout_rate)(out)
    
        return out


class RegNet(tf.keras.layers.Layer):

    def __init__(self, input_nodes=64, **kwargs):
        super(RegNet, self).__init__(**kwargs)
        self.input_nodes = input_nodes

    def __call__(self, inputs, mask=None):
        out = layers.BatchNormalization()(inputs)
        out = layers.Dense(self.input_nodes)(out)
        out = layers.LeakyReLU()(out)
        out = layers.Dense(self.input_nodes)(out)
        out = layers.LeakyReLU()(out)
        out = layers.Dense(16)(out)
        out = layers.LeakyReLU()(out)
        out = layers.Dense(1, activation="relu", dtype='float32')(out)
        return out

class ClassNet(tf.keras.layers.Layer):

    def __init__(self, input_nodes=64, **kwargs):
        super(ClassNet, self).__init__(**kwargs)
        self.input_nodes = input_nodes

    def __call__(self, inputs, mask=None):
        out = layers.BatchNormalization()(inputs)
        out = layers.Dense(self.input_nodes)(out)
        out = layers.LeakyReLU()(out)
        out = layers.Dense(self.input_nodes)(out)
        out = layers.LeakyReLU()(out)
        out = layers.Dense(16)(out)
        out = layers.LeakyReLU()(out)
        out = layers.Dense(5, activation="softmax", dtype='float32')(out)
        return out

class OutNet(tf.keras.layers.Layer):

    def __init__(self, input_nodes=64, nets=[], **kwargs):
        super(OutNet, self).__init__(**kwargs)
        self.input_nodes = input_nodes
        self.nets = nets

    def __call__(self, inputs, mask=None):
        output = []
        for i, net in enumerate(self.nets):
            if i == 0:
                output = [net(inputs)]
            else:
                output.append(net(inputs))
        return output