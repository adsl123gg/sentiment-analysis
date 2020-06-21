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
model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size,
                           output_dim=embedding_dim,
                           input_length=maxlen))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dense(10))
model.add(layers.Dense(1, activation="sigmoid"))


# train_model(xtrain, xtest, y_train, y_test, maxlen, model, epochs=15)
train_model(xtrain, xtest, y_train, y_test, maxlen, model, epochs=30, real_xtest=real_xtest)

