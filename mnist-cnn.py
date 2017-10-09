from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import tensorflow as tf 


# model the CNN
def cnn_model(features, labels, mode):

    # The model architecture is Conv->Relu->Pool->Conv->Relu->Pool->FC->FC

    # First layer is the input layer
    # model the layer into a 4-d tensor (batch_size, width, height, channels)
    # mnist dataset has 28*28 monochromatic images
    # keep batch size as 1 so that network treats it as a hyperparameter
    # and finds the optimum value
    input_layer = tf.reshape(features['x'], [-1,28,28,1])

    # Conv layer 1
    # 5*5 filters, output = (28-5+2P)/s + 1. we add padding of 2 to retain height/width (28-5+4)/1 + 1 = 28
    # we use 32 filters, hence k=32
    # output tensor shape: [batch_size, 28,28,32]
    conv_layer1 = tf.layers.conv2d(
        inputs= input_layer,
        filters= 32,
        strides=(1, 1),
        kernel_size= [5,5],
        padding= 'same',  # automatically pads to keep image size same
        activation= tf.nn.relu
    )

    # Pooling layer 1
    # use stride of 2, input: [batch_size,28,28,32]
    # output: [batch_size,14,14,32], since it pools every element in a 2*2 space
    # important, we are using max pooling here since it gives best result on average
    pool_layer1 = tf.layers.max_pooling2d(
        inputs= conv_layer1,
        strides= 2,
        pool_size= [2,2]
    )

    # Conv layer 2
    # k = 64, s = [5*5], padding = same
    # input = [batch_size, 14, 14, 32]
    # output = [batch_size, 14, 14, 64]
    conv_layer2 = tf.layers.conv2d(
        inputs= pool_layer1,
        filters= 64,
        kernel_size= [5,5],
        padding='same',
        activation=tf.nn.relu
    )

    # Pooling layer 2
    # continue to use stride of 2
    # input tensor= [batch_size, 14, 14, 64]
    # output tensor = [batch_size, 7, 7, 64]
    pool_layer2 = tf.layers.max_pooling2d(
        inputs= conv_layer2,
        pool_size= [2,2],
        strides= 2
    )

    # Finally, we have 2 fully connected layers
    
    # first flatten the output of the pool layer
    # input = [batch_size, 7, 7, 64]
    # output = [batch_size, 7*7*64]
    pool_layer2_flatten = tf.reshape(pool_layer2, [-1, 7*7*64])

    # First fully connected layer
    # densely connected layer with 1024
    # input tensor: [batch_size, 7*7*64]
    # output tensor: [batch_size, 1024]
    fc_layer1 = tf.layers.dense(
        inputs= pool_layer2_flatten,
        units= 1024,
        activation= tf.nn.relu
    )

    # add a dropout rate of 0.4, hence a 0.6 prob that element will be kept
    dropout = tf.layers.dropout(
        inputs= fc_layer1,
        rate= 0.4,
        training= mode == tf.estimator.ModeKeys.TRAIN
    )

    # Output layer
    # input = [batch_size, 1024]
    # output = [batch_size, 10]
    logits = tf.layers.dense(
        inputs= dropout, 
        units= 10
    )

    predictions = {
        'classes': tf.argmax(input = logits, axis = 1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    # if mode == predict, then use 
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # Calculate Loss for both the TRAIN and EVAL modes
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth = 10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels= onehot_labels,
        logits= logits
    )

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss= loss,
            global_step= tf.train.get_global_step()
        )
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, 
            predictions= predictions['classes']
        )
    }

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

tf.logging.set_verbosity(tf.logging.INFO)

def main(argv):
   # Load training and eval data
   mnist = tf.contrib.learn.datasets.load_dataset("mnist")
   train_data = mnist.train.images   # return a numpy array 
   train_labels = np.asarray(mnist.train.labels, dtype= np.int32)
   eval_data = mnist.test.images
   eval_labels = np.asarray(mnist.test.labels, dtype= np.int32)

   # Create estimator
   """ TODO: add model dir?? """
   mnist_classifier = tf.estimator.Estimator(
       model_fn = cnn_model,
       model_dir=None   )

   # set up logging for predictions
   # Log values in softmax tensor as label 'probabilities'
   tensor_to_log = {'probabilities': 'softmax_tensor'}
   logging_hook = tf.train.LoggingTensorHook(
       tensors= tensor_to_log, every_n_iter=50
   )

   # Train model
   train_input = tf.estimator.inputs.numpy_input_fn(
       x={'x': train_data},
       y = train_labels,
       batch_size=100,
       num_epochs=None,
       shuffle=True
   )
   mnist_classifier.train(
       input_fn= train_input,
       steps= 8000,
       hooks= [logging_hook]
   )

   # Evaluate Model and print results
   eval_input = tf.estimator.inputs.numpy_input_fn(
       x={'x': eval_data},
       y = eval_labels,
       num_epochs=1,
       shuffle=False
   )
   eval_results = mnist_classifier.evaluate(input_fn=eval_input)

   print(eval_results)

if __name__ == '__main__':
    tf.app.run()
