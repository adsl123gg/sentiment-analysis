from keras.models import Sequential
from keras import layers
import os
from sentiment.cross_base import train_model, get_train_test_df


os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

file = '../datasets/sentiment_train.csv'

xtrain, xtest, y_train, y_test, real_xtest, vocab_size = get_train_test_df(file)
maxlen = len(max((xtrain + xtest), key=len))
maxlen = 100

embedding_dim = 128
model = Sequential([
    layers.Embedding(input_dim=vocab_size,
                     output_dim=embedding_dim,
                     input_length=maxlen),
    layers.Conv1D(filters=50, kernel_size=5, strides=1, padding='valid'),
    layers.MaxPool1D(2, padding='valid'),
    layers.GlobalAveragePooling1D(),
    layers.Flatten(),
    layers.Dense(10, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

train_model(xtrain, xtest, y_train, y_test, maxlen, model, epochs=30)
# train_model(xtrain, xtest, y_train, y_test, maxlen, model, epochs=30, real_xtest=real_xtest)

