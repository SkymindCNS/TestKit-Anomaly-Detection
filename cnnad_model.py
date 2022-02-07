from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense,UpSampling2D
from keras.callbacks import TensorBoard
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras import optimizers
from tensorflow.keras.optimizers import Adam
import mlflow
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
from data_loader import generate_data
#input and encoding architecture

def model_init():
    input_layer = Input(shape=(412,412,3))
    encoded_layer1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    encoded_layer1 = MaxPooling2D( (2, 2), padding='same')(encoded_layer1)
    encoded_layer2 = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded_layer1)
    encoded_layer2 = MaxPooling2D( (2, 2), padding='same')(encoded_layer2)
    encoded_layer3 = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded_layer2)
    latent_view    = MaxPooling2D( (2, 2), padding='same')(encoded_layer3)

    # decoding architecture
    decoded_layer1 = Conv2D(16, (3, 3), activation='relu', padding='same')(latent_view)
    decoded_layer1 = UpSampling2D((2, 2))(decoded_layer1)
    decoded_layer2 = Conv2D(32, (3, 3), activation='relu', padding='same')(decoded_layer1)
    decoded_layer2 = UpSampling2D((2, 2))(decoded_layer2)
    decoded_layer3 = Conv2D(64, (3, 3), activation='relu')(decoded_layer2)
    decoded_layer3 = UpSampling2D((2, 2))(decoded_layer3)
    output_layer   = Conv2D(3, (3, 3), padding='same', activation='sigmoid')(decoded_layer3)

    # compile the model
    model = Model(input_layer, output_layer)
    model.compile(metrics=['accuracy'],optimizer='adam', loss='mse')
    return model

#tracking function
def track_performance_metrics(accuracy, precision, recall, f1_micro, f1_macro, training_time, prediction_time):
    # performance test set
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('f1_macro', f1_macro)
    mlflow.log_metric('f1_micro', f1_micro)
    mlflow.log_metric('precision', precision)
    mlflow.log_metric('recall', recall)

    mlflow.log_metric('training_time', training_time)
    mlflow.log_metric('prediction_time', prediction_time)

def evaluate_performance(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_micro = f1_score(y_test, y_pred, average='micro')
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
        
    return accuracy, precision, recall, f1_micro, f1_macro
# Fit and Predict Functions
def fit_keras(model, X_train, y_train, epochs=5):
    model.fit(X_train, y_train, epochs= epochs)
    return model

def predict_keras(model, X):
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred, axis=1)
    return y_pred

def perform_experiment(model, framework, X_train, X_val, y_train, y_val):
    # training model
    start_training = time.time()
    model = fit_keras(model, X_train, y_train, epochs=5)
    end_training = time.time()

    # predict the test set
    start_prediction = time.time()
    y_pred = predict_keras(model, X_val)
    end_prediction = time.time()

    # evaluate performance
    accuracy, precision, recall, f1_micro, f1_macro = evaluate_performance(y_val, y_pred)

    training_time = end_training - start_training
    prediction_time = end_prediction - start_prediction
    
    # Track performance metrics
    track_performance_metrics(accuracy, precision, recall, f1_micro, f1_macro, training_time, prediction_time)

if __name__ == "__main__":
    tmp_training_base_path = 'train_dataset'
    tmp_test_base_path='validation_dataset'
    anomoly_base_path='anomaly-dataset'
    #es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    #model_name = 'autoencoder_model.{epoch:03d}.h5'
    #save_best = keras.callbacks.ModelCheckpoint(model_name, monitor='val_loss', save_best_only=True, mode='min')
    train_generator,validation_generator,anomaly_generator =generate_data(tmp_training_base_path,tmp_test_base_path,anomoly_base_path,416,32)
    model = model_init()
    model.summary()
    history = model.fit(train_generator,validation_data=validation_generator, epochs=30)#callbacks=[es, save_best])
    model.save('autoencoder.h5')
    plt.plot(history.history['loss'])               
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('MSLE Loss')
    plt.legend(['loss', 'val_loss'])
    plt.show()