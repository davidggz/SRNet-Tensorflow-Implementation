import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def conv_layer(input_tensor, num_filters, kernel_size, strides, padding='same'):
    
    # He initializer
    filter_initializer = tf.keras.initializers.HeNormal()

    # Bias initializer
    bias_initializer = tf.keras.initializers.Constant(value=0.2)

    # L2 regularization for the filters
    filter_regularizer = tf.keras.regularizers.L2(l2=2e-4)
    
    x = layers.Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding=padding,
                  kernel_initializer=filter_initializer,
                  bias_initializer=bias_initializer,
                  kernel_regularizer=filter_regularizer,
                  use_bias=True)(input_tensor)
    
    return x


def layer_T1(input_tensor, num_filters):
    # Convolutional layer
    x = conv_layer(input_tensor, 
                   num_filters=num_filters, 
                   kernel_size=(3, 3), 
                   strides=1)
    
    # Batch normalization layer
    x = layers.BatchNormalization(momentum=0.9)(x)

    # ReLU activation layer
    x = layers.ReLU()(x)
    
    return x


def layer_T2(input_tensor, num_filters):
    # Add the layer T1 to the beginning of Layer T2
    x = layer_T1(input_tensor, num_filters)
    
    # Convolutional layer
    x = conv_layer(x, 
                   num_filters=num_filters, 
                   kernel_size=(3, 3), 
                   strides=1)
    
    # Batch normalization layer
    x = layers.BatchNormalization(momentum=0.9)(x)
    
    # Create the residual connection
    x = layers.add([input_tensor, x]) 
    
    return x


def layer_T3(input_tensor, num_filters):
    # MAIN BRANCH
    # Add the layer T1 to the beginning of Layer T2
    x = layer_T1(input_tensor, num_filters)
    
    # Convolutional layer
    x = conv_layer(x, 
                   num_filters=num_filters, 
                   kernel_size=(3, 3), 
                   strides=1)
    
    # Batch normalization layer
    x = layers.BatchNormalization(momentum=0.9)(x)
    
    # Average pooling layer
    x = layers.AveragePooling2D(pool_size=(3, 3), 
                                strides=2,
                                padding='same')(x)
    
    # SECONDARY BRANCH
    # Special convolutional layer. 
    y = conv_layer(input_tensor, 
                   num_filters=num_filters, 
                   kernel_size=(1, 1), 
                   strides=2)
    
    # Batch normalization layer
    y = layers.BatchNormalization(momentum=0.9)(y)
    
    # Create the residual connection
    output = layers.add([x, y]) 
    
    return output


def layer_T4(input_tensor, num_filters):
    # Add the layer T1 to the beginning of Layer T2
    x = layer_T1(input_tensor, num_filters)
    
    # Convolutional layer
    x = conv_layer(x, 
                   num_filters=num_filters, 
                   kernel_size=(3, 3), 
                   strides=1)
    
    # Batch normalization layer
    x = layers.BatchNormalization(momentum=0.9)(x)
    
    # Global Average Pooling layer
    x = layers.GlobalAveragePooling2D()(x)
    
    return x


def fully_connected(input_tensor):
    
    # Dense weight initializer N(0, 0.01)
    dense_initializer = tf.random_normal_initializer(0, 0.01)
    
    # Bias initializer for the fully connected network
    bias_dense_initializer = tf.constant_initializer(0.)
    
    x = layers.Flatten()(input_tensor)
    x = layers.Dense(512, 
                     activation=None,
                     use_bias=False,
                     kernel_initializer=dense_initializer,
                     bias_initializer=bias_dense_initializer)(x)

        
    output = layers.Dense(1, activation='sigmoid')(x)
    
    return output


def create_SRNet(input_image_size):
    # The input layer has the shape (256, 256, 1)
    input_layer = layers.Input(shape=input_image_size)

    x = layer_T1(input_layer, 64)
    x = layer_T1(x, 16)
    
    x = layer_T2(x, 16)
    x = layer_T2(x, 16)
    x = layer_T2(x, 16)
    x = layer_T2(x, 16)
    x = layer_T2(x, 16)
    
    x = layer_T3(x, 16)
    x = layer_T3(x, 64)
    x = layer_T3(x, 128)
    x = layer_T3(x, 256)
    
    x = layer_T4(x, 512)
    
    output = fully_connected(x)
    
    model = Model(inputs=input_layer, outputs=output, name="SRNet")
    
    return model