from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from sklearn.model_selection import train_test_split
import pandas as pd
from functools import reduce
import os


def convert(rs):
    return 1 if rs == 'Positive' else 0


def reverse(label):
    return 'Positive' if label == 1 else 'Negative'


def statics(xtrain):
    len_map = {}
    for d in xtrain:
        if len(d) in len_map.keys():
            len_map[len(d)] = len_map[len(d)] + 1
        else:
            len_map[len(d)] = 1

    total_score = sum(len_map.values())
    prev_score = 0
    for x in sorted(len_map.keys()):
        prev_score += len_map[x]
        print(x, prev_score / total_score)


def get_train_test_df(file):
    df_train = pd.read_csv(file)
    df_train.columns = ["label", "text"]
    x = df_train['text'].values
    yy = df_train['label'].values

    y = list(map(convert, yy))

    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.1, random_state=123)
    # x_train = x
    # y_train = y

    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(x)
    xtrain = tokenizer.texts_to_sequences(x_train)
    xtest = tokenizer.texts_to_sequences(x_test)

    real_xtest = tokenizer.texts_to_sequences(load_test_data())

    vocab_size = len(tokenizer.word_index) + 1

    return xtrain, xtest, y_train, y_test, real_xtest, vocab_size


def train_model(xtrain, xtest, y_train, y_test, maxlen, model,
                epochs=20, batch_size=64, real_xtest=None):
    xtrain = pad_sequences(xtrain, maxlen=maxlen)
    xtest = pad_sequences(xtest, maxlen=maxlen)

    model.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=['accuracy'])
    # output model summary
    model.summary()
    model.fit(xtrain, y_train, epochs=epochs, batch_size=batch_size, verbose=2,
              # validation_split=0.1,
              validation_data=(xtest, y_test),
              )

    loss, acc = model.evaluate(xtrain, y_train, verbose=False)
    print("Training Accuracy: ", loss, acc)
    loss, acc = model.evaluate(xtest, y_test, verbose=False)
    print("Test Accuracy: ", loss, acc)

    if (real_xtest != None):
        real_xtest = pad_sequences(real_xtest, maxlen=maxlen)
        ypred = model.predict(real_xtest)
        ypred[ypred > 0.5] = 1
        ypred[ypred <= 0.5] = 0
        result = pd.DataFrame()
        # result['id'] = load_test_data()['id']
        result['label'] = list(map(reverse, ypred))
        print(result)
        result.to_csv("../datasets/rs.csv")
        # result = zip(ypred, load_test_data())
        # for i in result:
        #     print(i)


def load_test_data():
    df_test = pd.read_csv('../datasets/sentiment_test.csv')
    x_test = df_test['text'].values
    return x_test
