from keras.layers import BatchNormalization, Activation, Conv1D, MaxPooling1D, ZeroPadding1D, Input
from keras.models import Model
import numpy as np
import librosa
import glob


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


def build_model():
    model_weights = dict(np.load('models/sound8.npy', encoding="latin1", allow_pickle=True).item())
    seq_layer_parameters = [{'name': 'conv1', 'num_filters': 16, 'padding': 32,
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

    # Build sequential layers
    inputs = Input(shape=(None, 1))
    prev_layer = inputs
    for seq_layer_parameter in seq_layer_parameters:
        prev_layer = ZeroPadding1D(padding=seq_layer_parameter['padding'])(prev_layer)

        conv_layer = Conv1D(seq_layer_parameter['num_filters'],
                            kernel_size=seq_layer_parameter['kernel_size'],
                            strides=seq_layer_parameter['conv_strides'],
                            padding='valid',
                            name=seq_layer_parameter['name'])

        prev_layer = conv_layer(prev_layer)
        weights = model_weights[seq_layer_parameter['name']]['weights'].reshape(conv_layer.get_weights()[0].shape)
        biases = model_weights[seq_layer_parameter['name']]['biases']
        conv_layer.set_weights([weights, biases])

        gamma = model_weights[seq_layer_parameter['name']]['gamma']
        beta = model_weights[seq_layer_parameter['name']]['beta']
        mean = model_weights[seq_layer_parameter['name']]['mean']
        var = model_weights[seq_layer_parameter['name']]['var']
        batch_norm = BatchNormalization()
        prev_layer = batch_norm(prev_layer)
        batch_norm.set_weights([gamma, beta, mean, var])

        prev_layer = Activation('relu')(prev_layer)

        if 'pool_size' in seq_layer_parameter:
            prev_layer = MaxPooling1D(pool_size=seq_layer_parameter['pool_size'],
                                      strides=seq_layer_parameter['pool_strides'],
                                      padding='valid')(prev_layer)

    # Build split output layers
    object_output_layer = ZeroPadding1D(padding=0)(prev_layer)
    conv_layer = Conv1D(1000, kernel_size=8, strides=2, padding='valid', name='conv8')
    object_output_layer = conv_layer(object_output_layer)
    weights = model_weights['conv8']['weights'].reshape(conv_layer.get_weights()[0].shape)
    biases = model_weights['conv8']['biases']
    conv_layer.set_weights([weights, biases])

    scene_output_layer = ZeroPadding1D(padding=0)(prev_layer)
    conv_layer = Conv1D(401, kernel_size=8, strides=2, padding='valid', name='conv8_2')
    scene_output_layer = conv_layer(scene_output_layer)
    weights = model_weights['conv8_2']['weights'].reshape(conv_layer.get_weights()[0].shape)
    biases = model_weights['conv8_2']['biases']
    conv_layer.set_weights([weights, biases])

    return Model(inputs=inputs, outputs=[object_output_layer, scene_output_layer])


def predictions_to_categories(prediction, cat):
    scenes = []
    for p in range(prediction.shape[1]):
        index = np.argmax(prediction[0, p, :])
        scenes.append(cat[index])
    return scenes


if __name__ == '__main__':
    model_object = build_model()
    model_object.summary()
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
        plain_prediction = model_object.predict(file)
        object_pred_cat = predictions_to_categories(plain_prediction[0], object_categories)
        print("OBJECT {}: {}".format(filename, object_pred_cat))

        scene_pred_cat = predictions_to_categories(plain_prediction[1], scene_categories)
        print("SCENE {}: {}".format(filename, scene_pred_cat))
