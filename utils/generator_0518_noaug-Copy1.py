import tensorflow.keras
from tensorflow.keras import backend as K
from scipy.io import wavfile
import numpy as np
import librosa
import soundfile as sf
from tensorflow.keras.utils import to_categorical
import msgpack
import msgpack_numpy as m
import os
from tensorflow.keras.layers import Conv2D, maximum, add, SeparableConv2D

m.patch()

##################################################
#
#  DataGenerator_cqt(): Data Generator 
#
##################################################
#  Input:
#    list_IDs : A list of file names for the generator. each file is a sound file.
#    labels : A list of labels for list of IDs. 
#    data_dir : Data directory (ASVspoof2019)
#    batch_size : batch size
#    sr : sampling rate
#    pre_emphasis : pre_emphasis ratio
#    sec : adjusted seconds of the sample. 
#    filter_scale : Filter scale factor. Small values (<1) use shorter windows for improved time resolution.
#    n_bins : Number of frequency bins, starting at fmin 
#    fmin : Minimum frequency. Defaults to C1 ~= 32.70 Hz
#    is_flac : Are sound files in .flac format? 
#    shuffle: shuffle ids
#    ...
#

class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, data_dir, batch_size=32, 
                 sr = 16000, pre_emphasis = 0,feature = 'cqt',
                 sec=3.0, mono = 1,
#                 frame_size = 0.04, frame_stride = 0.02, 
                 n_classes=2, shuffle=True, is_flac = False, tofile = False,
                 filter_scale = 1, n_bins = 100, fmin = 10,
                 n_mels = 62, frame_length = 0.025,
                  hop_length = 128, win_length = 512, 
                 cutmix = False, cutout = False, specaug = False, specmix = False,
                 beta_param = False, lowpass = False, highpass = False, ranfilter = False, ranfilter2 = False, dropblock = False
                 ):
        'Initialization'
        self.data_dir = data_dir
        self.sec = sec
        self.sr = sr
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.mono = mono
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.pre_emph = pre_emphasis
        self.feature = feature
        self.is_flac = is_flac

        # self.frame_size = frame_size
        # self.frame_stride = frame_stride

        self.filter_scale = filter_scale
        self.n_bins = n_bins
        self.fmin = fmin
        self.is_flac = is_flac
        
        if self.feature =="cqt":
            Sig = feature_extract_cqt(list_IDs[0], samp_sec=sec, pre_emphasis = pre_emphasis,
                               filter_scale = filter_scale, n_bins = n_bins, fmin = fmin, is_flac = is_flac)[0]
            print("cqt")
            
        self.M = Sig.shape[0]
        self.N = Sig.shape[1]
        
        self.on_epoch_end()
        if tofile :
            self.tofile = tofile
            if feature == "cqt" :
                self.name = data_dir + 'tmp/' +str(self.tofile) + 'fm_'+str(self.fmin) + 'nb_'+str(self.n_bins)+'fs_' + str(self.filter_scale)+'_'
#+'.msgpack'
        else :
            self.tofile = False
            self.name = False
        
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        samp_sec = self.sec
        dim = int(samp_sec * self.sr)

        X = np.empty((self.batch_size, self.M, self.N, self.mono))
        y = np.empty((self.batch_size), dtype=int)

        #################################################
        # Generate data
        #################################################
        for j, ID in enumerate(list_IDs_temp):
            if self.name :
                write_fnm = self.name + ID.split('/')[6] + '.msgpack'
                if os.path.exists(write_fnm) : 
                    with open(write_fnm, 'rb') as data_file:
                        data_loaded = msgpack.unpack(data_file)
                    signal = msgpack.unpackb(data_loaded, object_hook=m.decode)
                else :
                    try :
                        if self.feature =="cqt":
                            signal = feature_extract_cqt(ID, samp_sec=self.sec,
                                      pre_emphasis = self.pre_emph,
                                      filter_scale = self.filter_scale, n_bins = self.n_bins, fmin = self.fmin, 
                                      is_flac = self.is_flac )
                    except :
                        print(self.data_dir+ID)
                        raise
                        
                    x_enc = msgpack.packb(signal, default=m.encode)
                    with open(write_fnm, 'wb') as outfile:
                        msgpack.pack(x_enc, outfile)
            else :
                if self.feature == "cqt":
                    signal = feature_extract_cqt(ID, samp_sec=self.sec,
                                      pre_emphasis = self.pre_emph,
                                      filter_scale = self.filter_scale, n_bins = self.n_bins, fmin = self.fmin, 
                                      is_flac = self.is_flac )
                    
            X[j,] = signal[0].reshape(self.M, self.N, self.mono)

            # Store class
            y[j] = self.labels[ID]
#        print(y.shape)

        return X, to_categorical(np.array(y), self.n_classes)



##################################################
#
#  feature_extract_cqt(): extract a CQT feature 
#
##################################################
#  Input:
#    fnm : a file name
#    samp_sec : adjusted seconds of the sample. 
#    pre_emphasis : pre_emphasis ratio
#    filter_scale : Filter scale factor. Small values (<1) use shorter windows for improved time resolution.
#    n_bins : Number of frequency bins, starting at fmin 
#    fmin : Minimum frequency. Defaults to C1 ~= 32.70 Hz
#    is_flac : Are sound files in .flac format? 
#
#  output:
#    Constant-Q value each frequency at each time.

def feature_extract_cqt(fnm, samp_sec=3, pre_emphasis = 0, filter_scale = 1, n_bins = 100, fmin = 10,
                     is_flac = True) :

#    if is_flac :
    data, sample_rate = sf.read(fnm, dtype = 'int16')
#        data = data * 32768 #normalizing flac features to match previous wav features
#    else:
#        sample_rate, data = wavfile.read(fnm)

    data = data * 1.0

    if len(data) > sample_rate * samp_sec : 
        n_samp = len(data) // int(sample_rate * samp_sec)
        signal = []
        for i in range(n_samp) :
            signal.append(data[ int(sample_rate * samp_sec)*i:(int(sample_rate * samp_sec)*(i+1))]) 
    else :
        n_samp = 1
        signal = np.zeros(int(sample_rate*samp_sec,))
        for i in range(int(sample_rate * samp_sec) // len(data)) :
            signal[(i)*len(data):(i+1)*len(data)] = data
        num_last = int(sample_rate * samp_sec) - len(data)*(i+1)
        signal[(i+1)*len(data):int(sample_rate * samp_sec)] = data[:num_last]
        signal = [signal]    
    
    
    Sig = []
    for i in range(n_samp) :
        if pre_emphasis :
            emphasized_signal = np.append(signal[i][0], signal[i][1:] - pre_emphasis * signal[i][:-1])
        else :
            emphasized_signal = signal[i] 
            
        Sig.append(np.log(np.abs(librosa.cqt(signal[i], sr=sample_rate,
                                             filter_scale = filter_scale, n_bins=n_bins,
                                             fmin = fmin))+1))
    return Sig





    
##################################################
#
#  init_input_shape(): Calculate input shape  
#
##################################################
#  Input:
#    fnm : a file name
#    samp_sec : adjusted seconds of the sample. 
#    pre_emphasis : pre_emphasis ratio
#    filter_scale : Filter scale factor. Small values (<1) use shorter windows for improved time resolution.
#    n_bins : Number of frequency bins, starting at fmin 
#    fmin : Minimum frequency. Defaults to C1 ~= 32.70 Hz
#    is_flac : Are sound files in .flac format? 
#
#  output:
#    input shape of the CQT feature


def init_input_shape(fnm, samp_sec = 3, pre_emphasis = 0, filter_scale = 1, n_bins=100,
                     fmin = 10, is_flac = False):
    Sig = feature_extract_cqt(fnm, samp_sec=samp_sec, pre_emphasis = pre_emphasis,
                           filter_scale = filter_scale, n_bins=n_bins, fmin = fmin,
                           is_flac = is_flac)
    M, N = Sig[0].shape

    if K.image_data_format() == 'channels_first':
        input_shape = (1, M, N)
    else:
        input_shape = (M, N, 1)
    return input_shape

