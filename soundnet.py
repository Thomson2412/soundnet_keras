from keras.layers import BatchNormalization, Activation, Conv1D, MaxPooling1D, ZeroPadding1D, InputLayer
from keras.models import Sequential
import numpy as np
import librosa
import glob
import tensorflow as tf


def preprocess(audio):
    audio *= 256.0  # SoundNet needs the range to be between -256 and 256
    # reshaping the audio data so it fits into the graph (batch_size, num_samples, num_filter_channels)
    audio = np.reshape(audio, (1, -1, 1))
    return audio


def load_audio(audio_file):
    sample_rate = 22050  # SoundNet works on mono audio files with a sample rate of 22050.
    audio, sr = librosa.load(audio_file, dtype='float32', sr=sample_rate, mono=True)
    audio = preprocess(audio)
    return audio


def build_model(cat_type):
    model_weights = dict(np.load('models/sound8.npy', encoding="latin1", allow_pickle=True).item())
    model_seq = Sequential()
    model_seq.add(InputLayer(batch_input_shape=(1, None, 1)))

    filter_parameters = [{'name': 'conv1', 'num_filters': 16, 'padding': 32,
                          'kernel_size': 64, 'conv_strides': 2,
                          'pool_size': 8, 'pool_strides': 8},

                         {'name': 'conv2', 'num_filters': 32, 'padding': 16,
                          'kernel_size': 32, 'conv_strides': 2,
                          'pool_size': 8, 'pool_strides': 8},

                         {'name': 'conv3', 'num_filters': 64, 'padding': 8,
                          'kernel_size': 16, 'conv_strides': 2},

                         {'name': 'conv4', 'num_filters': 128, 'padding': 4,
                          'kernel_size': 8, 'conv_strides': 2},

                         {'name': 'conv5', 'num_filters': 256, 'padding': 2,
                          'kernel_size': 4, 'conv_strides': 2,
                          'pool_size': 4, 'pool_strides': 4},

                         {'name': 'conv6', 'num_filters': 512, 'padding': 2,
                          'kernel_size': 4, 'conv_strides': 2},

                         {'name': 'conv7', 'num_filters': 1024, 'padding': 2,
                          'kernel_size': 4, 'conv_strides': 2}
                         ]
    if cat_type == "object":
        filter_parameters.append(
            {'name': 'conv8', 'num_filters': 1000, 'padding': 0, 'kernel_size': 8, 'conv_strides': 2})
    elif cat_type == "scene":
        filter_parameters.append(
            {'name': 'conv8_2', 'num_filters': 401, 'padding': 0, 'kernel_size': 8, 'conv_strides': 2})
    else:
        raise ValueError("Type must be scene or object")

    for filter_parameter in filter_parameters:
        model_seq.add(ZeroPadding1D(padding=filter_parameter['padding']))
        model_seq.add(Conv1D(filter_parameter['num_filters'],
                             kernel_size=filter_parameter['kernel_size'],
                             strides=filter_parameter['conv_strides'],
                             padding='valid'))
        weights = model_weights[filter_parameter['name']]['weights'].reshape(
            model_seq.layers[-1].get_weights()[0].shape)
        biases = model_weights[filter_parameter['name']]['biases']

        model_seq.layers[-1].set_weights([weights, biases])

        if 'conv8' not in filter_parameter['name']:
            gamma = model_weights[filter_parameter['name']]['gamma']
            beta = model_weights[filter_parameter['name']]['beta']
            mean = model_weights[filter_parameter['name']]['mean']
            var = model_weights[filter_parameter['name']]['var']

            model_seq.add(BatchNormalization())
            model_seq.layers[-1].set_weights([gamma, beta, mean, var])
            model_seq.add(Activation('relu'))
        if 'pool_size' in filter_parameter:
            model_seq.add(MaxPooling1D(pool_size=filter_parameter['pool_size'],
                                       strides=filter_parameter['pool_strides'],
                                       padding='valid'))

    return model_seq


def predictions_to_categories(prediction, cat):
    scenes = []
    for p in range(prediction.shape[1]):
        index = np.argmax(prediction[0, p, :])
        scenes.append(cat[index])
    return scenes


if __name__ == '__main__':
    print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
    model_object = build_model("object")
    model_scene = build_model("scene")
    with open('categories/categories_imagenet.txt', 'r') as f:
        object_categories = f.read().split('\n')
    with open('categories/categories_places2.txt', 'r') as f:
        scene_categories = f.read().split('\n')

    files = {}
    for filename in sorted(glob.glob("sounds/**/*.wav")):
        duration = librosa.get_duration(filename=filename)
        if duration >= 5.1:
            print(duration)
            files[filename] = load_audio(filename)
            if len(files) > 20:
                break

    for filename, file in files.items():
        plain_object_prediction = model_object.predict(file)
        object_pred_cat = predictions_to_categories(plain_object_prediction, object_categories)
        print("OBJECT {}: {}".format(filename, object_pred_cat))

        plain_scene_prediction = model_scene.predict(file)
        scene_pred_cat = predictions_to_categories(plain_scene_prediction, scene_categories)
        print("SCENE {}: {}".format(filename, scene_pred_cat))
