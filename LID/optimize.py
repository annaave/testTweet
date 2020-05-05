#from: https://medium.com/@crawftv/parameter-hyperparameter-tuning-with-bayesian-optimization-7acf42d348e1
#https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/19_Hyper-Parameters.ipynb
#imports we know we'll need
import skopt
# !pip install scikit-optimize if  necessary
from skopt import gbrt_minimize, gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from keras.optimizers import Adam

dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform',
                         name='learning_rate')
dim_num_dense_layers = Integer(low=1, high=5, name='num_dense_layers')
dim_num_input_nodes = Integer(low=1, high=512, name='num_input_nodes')
dim_num_dense_nodes = Integer(low=1, high=28, name='num_dense_nodes')
dim_activation = Categorical(categories=['relu', 'sigmoid'],
                             name='activation')
dim_batch_size = Integer(low=1, high=128, name='batch_size')
dim_adam_decay = Real(low=1e-6, high=1e-2, name="adam_decay")

dimensions = [dim_learning_rate,
              dim_num_dense_layers,
              dim_num_input_nodes,
              dim_num_dense_nodes,
              dim_activation,
              dim_batch_size,
              dim_adam_decay
             ]
default_parameters = [1e-3, 1, 128, 13, 'relu', 32, 1e-3]


def create_model(learning_rate, num_dense_layers, num_input_nodes,
                 num_dense_nodes, activation, adam_decay):
    # start the model making process and create our first layer
    model = Sequential()
    model.add(Dense(num_input_nodes, input_shape=input_shape, activation=activation
                    ))
    # create a loop making a new dense layer for the amount passed to this model.
    # naming the layers helps avoid tensorflow error deep in the stack trace.
    for i in range(num_dense_layers):
        name = 'layer_dense_{0}'.format(i + 1)
        model.add(Dense(num_dense_nodes,
                        activation=activation,
                        name=name
                        ))
    # add our classification layer.
    model.add(Dense(10, activation='softmax'))

    # setup our optimizer and compile
    adam = Adam(lr=learning_rate, decay=adam_decay)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers, num_input_nodes,
            num_dense_nodes, activation, batch_size, adam_decay):
    model = create_model(learning_rate=learning_rate,
                         num_dense_layers=num_dense_layers,
                         num_input_nodes=num_input_nodes,
                         num_dense_nodes=num_dense_nodes,
                         activation=activation,
                         adam_decay=adam_decay
                         )

    # named blackbox becuase it represents the structure
    blackbox = model.fit(x=x_train_pad,
                         y=y_train,
                         epochs=3,
                         batch_size=batch_size,
                         validation_split=0.15,
                         )
    # return the validation accuracy for the last epoch.
    accuracy = blackbox.history['val_acc'][-1]

    # Print the classification accuracy.
    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    print()

    # Delete the Keras model with these hyper-parameters from memory.
    del model

    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    K.clear_session()
    ops.reset_default_graph()

    # the optimizer aims for the lowest score, so we return our negative accuracy
    return -accuracy


K.clear_session()
ops.reset_default_graph()

gp_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            n_calls=12,
                            noise= 0.01,
                            n_jobs=-1,
                            kappa = 5,
                            x0=default_parameters)

model = create_model(gp_result.x[0],gp_result.x[1],gp_result.x[2],gp_result.x[3],gp_result.x[4],gp_result.x[5])
model.fit(x_train_pad,y_train, epochs=3)
model.evaluate(x_test_pad,y_test)


#https://medium.com/@crawftv/parameter-hyperparameter-tuning-with-bayesian-optimization-7acf42d348e1
#https://towardsdatascience.com/finding-the-right-architecture-for-neural-network-b0439efa4587
