import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
import argparse
import nsml
from nsml import DATASET_PATH

print('tf version: ', tf.__version__)

def get_index(data):
    return data.split("_")[1]

def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis = -1)

def get_waveform_and_label(left_file_path, right_file_path, label):
    left_audio_binary = tf.io.read_file(left_file_path)
    left_waveform = decode_audio(left_audio_binary)
    
    right_audio_binary = tf.io.read_file(right_file_path)
    right_waveform = decode_audio(right_audio_binary)
    
    label = label
    return left_waveform, right_waveform, label

def test_get_waveform(left_file_path, right_file_path):
    left_audio_binary = tf.io.read_file(left_file_path)
    left_waveform = decode_audio(left_audio_binary)
    
    right_audio_binary = tf.io.read_file(right_file_path)
    right_waveform = decode_audio(right_audio_binary)
    
    return left_waveform, right_waveform

def get_spectrogram(waveform):
    max = 370000

    zero_padding = tf.zeros([max] - tf.shape(waveform), dtype = tf.float32)
    
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    
    return spectrogram

def get_spectrogram_(file_path):
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    spectrogram = get_spectrogram(waveform)

    spectrogram = tf.expand_dims(spectrogram, -1)
    
    return spectrogram


def get_spectrogram_and_label(left_audio, right_audio, label):
    left_spectrogram = get_spectrogram(left_audio)
    right_spectrogram = get_spectrogram(right_audio)
    
    left_spectrogram = tf.expand_dims(left_spectrogram, -1)
    right_spectrogram = tf.expand_dims(right_spectrogram, -1)
    return (left_spectrogram, right_spectrogram), label

def test_get_spectrogram(left_audio, right_audio):
    left_spectrogram = get_spectrogram(left_audio)
    right_spectrogram = get_spectrogram(right_audio)
    
    left_spectrogram = tf.expand_dims(left_spectrogram, -1)
    right_spectrogram = tf.expand_dims(right_spectrogram, -1)
    return left_spectrogram, right_spectrogram

def train_label_loader(root_path) :
    train_path = os.path.join(root_path, 'train')
    train_label = pd.read_csv(train_path + '/train_label')
    
    return train_label


def make_model(train_left, train_right) :

    left_input = layers.Input(shape=train_left[0].shape)
    right_input = layers.Input(shape=train_right[0].shape)

    final_input = layers.concatenate([left_input, right_input])

    resize = preprocessing.Resizing(500, 129)(final_input)
    layer1 = layers.Conv2D(32, 3, activation = "relu")(resize)
    layer2 = layers.Conv2D(64, 3, activation = "relu")(layer1)
    pool1 = layers.MaxPooling2D()(layer2)
    drop1 = layers.Dropout(0.25)(pool1)
    flat = layers.Flatten()(drop1)
    dense1 = layers.Dense(64, activation = "relu")(flat)
    dense2 = layers.Dense(32, activation = "relu")(flat)
    output = layers.Dense(2, activation = "sigmoid")(dense2)

    model = tf.keras.models.Model(inputs = [left_input, right_input], outputs = output)
    optimizer = tf.keras.optimizers.Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ["accuracy"])

    return model

def add_paths_col(tmp, root_path = DATASET_PATH + "/train/train_data/") :
   
    tmp["left_paths"] = root_path + tmp["file_name"]
    tmp["right_paths"] = root_path + tmp["file_name_"]

    return tmp


def preprocess(spectrogram_ds) :
    ds = spectrogram_ds
    ds = ds.batch(batch_size)

    return ds


def test_preprocess(tmp, root_path) :
    tmp["left_paths"] = root_path + tmp["file_name"]
    tmp["right_paths"] = root_path + tmp["file_name_"]

    left_paths = np.array(tmp["left_paths"])
    right_paths = np.array(tmp["right_paths"])
    
    file_ds = tf.data.Dataset.from_tensor_slices((left_paths, right_paths, label))
    waveform_ds = file_ds.map(get_waveform_and_label, num_parallel_calls = AUTOTUNE)
    spectrogram_ds = waveform_ds.map(get_spectrogram_and_label, num_parallel_calls=AUTOTUNE)
    
    ds = spectrogram_ds
    ds = ds.batch(batch_size)

    return ds


def bind_model(parser):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *parser):
        save_dir = os.path.join(dir_name, 'model')
        model.save(save_dir)

        print("저장 완료!")

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *parser):
        save_dir = os.path.join(dir_name, 'model')
        global model
        model = tf.keras.models.load_model(save_dir)

        print("로딩 완료!")

    def infer(test_path, **kwparser):
        test_root_path = test_path + '/test_data/wav/'
        test_tmp = pd.read_csv(test_path + '/test_data/test_data')
        
        test_tmp = add_paths_col(test_tmp, root_path=test_root_path) #dataframe

        left_paths = np.array(test_tmp["left_paths"])
        right_paths = np.array(test_tmp["right_paths"])
        
        test_left = []
        test_right = []

        for i in left_paths:
            test_left.append(get_spectrogram_(i))
        for i in right_paths:
            test_right.append(get_spectrogram_(i))

        pred = np.argmax(model.predict([np.array(test_left), np.array(test_right)]), axis = 1)

        prob = [1]*len(pred)
        
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 
        # 리더보드 결과에 확률의 값은 영향을 미치지 않습니다
        return list(zip(prob, pred))

    nsml.bind(save=save, load=load, infer=infer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='nia_test')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--iteration', type=str, default='0')
    parser.add_argument('--pause', type=int, default=0)
    args = parser.parse_args()

    batch_size = 64
    AUTOTUNE = tf.data.AUTOTUNE

    if args.mode == 'train' :
        tmp = train_label_loader(DATASET_PATH)
        split_num = int(len(tmp) * 0.9)
        tmp = tmp.iloc[-split_num:]
        tmp = add_paths_col(tmp)

        left_paths = np.array(tmp["left_paths"])
        right_paths = np.array(tmp["right_paths"])
        label = np.array(tmp["label"])
        label = tf.keras.utils.to_categorical(label)

        train_left = []
        train_right = []

        for i in left_paths:
            train_left.append(get_spectrogram_(i))
        for i in right_paths:
            train_right.append(get_spectrogram_(i))

        model = make_model(train_left, train_right)
        
        bind_model(parser=args)

        if args.pause :
            nsml.paused(scope=locals())

        train_x = [np.array(train_left), np.array(train_right)]
        for epoch in range(args.epochs) :
            history = model.fit(x = train_x, y = label, epochs = 1)
            print('history: ', history.history)

            nsml.save(epoch)

    else :
        bind_model(parser=args) #load
        if args.pause :
            nsml.paused(scope=locals())

