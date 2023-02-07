import datetime
import tensorflow as tf
import numpy as np
import os

from utils.DataLoader import data_loader
from utils.Generator0 import DataGenerator, feature_extract_cqt, evalEER,  evalScore, evalEER_f, evalEER_f2, gen_fname
from models.models import get_ResMax, get_LCNN, sigmoidal_decay
from models.models2 import get_BCResMax 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import add,concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, maximum, DepthwiseConv2D, AveragePooling2D, UpSampling2D
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, SpatialDropout2D
from tensorflow.keras.layers import Convolution2D, GlobalAveragePooling2D, MaxPool2D, ZeroPadding2D
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.activations import relu, softmax, swish
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from sklearn.metrics import roc_curve

import pickle

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    except RuntimeError as e:
        print(e)

add2022 = '/home/ubuntu/data/ADD/'
asv2019 = '/home/ubuntu/data/asv2019/'

pathset = { 'add2022' : add2022 , 'asv2019':asv2019}
        
dl = data_loader(pathset)

#dl.get_data(data_pick = '2', tde_pick = 't', pl_pick = 'l', to = 't')
#dl.get_data(data_pick = '2', tde_pick = 'd', pl_pick = 'l', to = 't')
#dl.get_data(data_pick = '2', tde_pick = 'e', pl_pick = 'l', to = 'd')

mname = 'ResMax_ADD_'
datapick = '1' ## 1:ADD, 2:LA

dl.get_data(data_pick = datapick, tde_pick = 't', to = 't')
dl.get_data(data_pick = datapick, tde_pick = 'd', to = 'd')
dl.get_data(data_pick = datapick, tde_pick = 'e', to = 'e')

#track1 = data_loader(pathset)
#track1_generator = DataGenerator(track1.eval, track1.labels, **params_no_shuffle)



##  'ffmfa': [[3, 6], 6, "step", [-6,6] ],

for i in range(3) :
    sr = 16000
    sec = 9.0
    batch_size = 16
    feature = "cqt"
    #hop_length = 128
    #win_length = 512
    #n_bins = 80
    filter_scale = 1
    n_bins = 100
    fmin = 5
    epoch = 70
    beta_param = 0.7
    dropout_rate = 0.5 #np.random.choice([0.4, 0.5, 0.6])
    human_weight = 5.0 #np.random.choice([4.0, 5.0, 6.0])
    
    tmp_string = "tmp"
    params = {'sr': sr,
          'batch_size': batch_size,
          'feature': feature,
          'n_classes': 2,
          'sec': sec,
          'filter_scale': filter_scale,
          'fmin' : fmin,
          'n_bins': int(n_bins),
          'tofile': tmp_string,
          'shuffle': True,
          'beta_param': beta_param,
              'ffmfa': [[3, 6], 6, "step", [-6,6] ],
#              'specmix': False,    
#              'cutout': False,
#              'cutmix': False,
          'data_dir': asv2019
#          'lowpass': lowpass,
#          'highpass': highpass,
          #          'ranfilter' : [10,11,12,13,14,15]
#          'ranfilter2' : ranfilter2 
          #           'dropblock' : [30, 100]
          #'device' : device
    }
    params_no_shuffle = {'sr': sr,
                     'batch_size': batch_size,
                     'feature': feature,
                     'n_classes': 2,
                     'sec': sec,
                     'filter_scale': filter_scale,
                     'fmin' : fmin,
                     'n_bins': int(n_bins),
                     'tofile': tmp_string,
                     'shuffle': False,
                    'data_dir': asv2019
                     #'device': device
    }
    training_generator = DataGenerator(dl.train, dl.labels, **params)
    validation_generator = DataGenerator(dl.dev, dl.labels, **params_no_shuffle)
    eval_generator = DataGenerator(dl.eval, dl.labels, **params_no_shuffle)
    input_shape = training_generator.get_input_shape()
    model = get_ResMax(input_shape)
#    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['AUC'])

    EPOCHS = epoch
    lr = LearningRateScheduler(lambda e: sigmoidal_decay(e, end=EPOCHS))
    class_weight = {0: human_weight, 1: 1.} ## human: 0, 1: speaker
    history = model.fit_generator(generator=training_generator, #validation_data=track1_generator,
                              epochs=EPOCHS, class_weight=class_weight, callbacks=[lr], verbose=1)
    eer_val = evalEER(validation_generator, model)
    eer_eval = evalEER(eval_generator, model)
    endtxt1 = str(eer_val)[:6] + '.hdf5'
    savefnm = gen_fname(model_name = mname, params = params , dropout_rate = str(dropout_rate), human_weight = str(human_weight), endtxt = endtxt1 )
    savefnm = 'saved_models/' + savefnm
    model.save(savefnm)
    endtxt1 = str(eer_val)[:6] + '.npy'
    fnm = gen_fname(model_name = mname, params = params , dropout_rate = str(dropout_rate), human_weight = str(human_weight), endtxt = endtxt1)
    fnm = 'saved_results/' + fnm
    sc1 = model.predict(eval_generator)
    np.save(fnm, sc1)
    params_save = params
    params_save['model'] = mname
    params_save['human_weight'] = str(human_weight)
    params_save['dropout_rate'] = str(dropout_rate)
    params_save['eer_val'] = str(eer_val)[:6]
    params_save['eer_eval'] = str(eer_eval)[:6]
    params_save['saved_model'] = savefnm
    tnow = datetime.datetime.now()
    params_save['tnow'] = str(tnow)
    print(params_save)
    fnm = 'res_fa/rec'+str(tnow) + '.pickle'
    with open(fnm, 'wb') as f:
        pickle.dump(params_save, f, pickle.HIGHEST_PROTOCOL)

##  'ffmfa': [[3, 6], 6, "linear", [-6,6] ],

for i in range(3) :
    sr = 16000
    sec = 9.0
    batch_size = 16
    feature = "cqt"
    #hop_length = 128
    #win_length = 512
    #n_bins = 80
    filter_scale = 1
    n_bins = 100
    fmin = 5
    epoch = 70
    beta_param = 0.7
    dropout_rate = 0.5 #np.random.choice([0.4, 0.5, 0.6])
    human_weight = 5.0 #np.random.choice([4.0, 5.0, 6.0])
    
    tmp_string = "tmp"
    params = {'sr': sr,
          'batch_size': batch_size,
          'feature': feature,
          'n_classes': 2,
          'sec': sec,
          'filter_scale': filter_scale,
          'fmin' : fmin,
          'n_bins': int(n_bins),
          'tofile': tmp_string,
          'shuffle': True,
          'beta_param': beta_param,
              'ffmfa': [[3, 6], 6, "linear", [-6,6] ],
#              'specmix': False,    
#              'cutout': False,
#              'cutmix': False,
          'data_dir': asv2019
#          'lowpass': lowpass,
#          'highpass': highpass,
          #          'ranfilter' : [10,11,12,13,14,15]
#          'ranfilter2' : ranfilter2 
          #           'dropblock' : [30, 100]
          #'device' : device
    }
    params_no_shuffle = {'sr': sr,
                     'batch_size': batch_size,
                     'feature': feature,
                     'n_classes': 2,
                     'sec': sec,
                     'filter_scale': filter_scale,
                     'fmin' : fmin,
                     'n_bins': int(n_bins),
                     'tofile': tmp_string,
                     'shuffle': False,
                    'data_dir': asv2019
                     #'device': device
    }
    training_generator = DataGenerator(dl.train, dl.labels, **params)
    validation_generator = DataGenerator(dl.dev, dl.labels, **params_no_shuffle)
    eval_generator = DataGenerator(dl.eval, dl.labels, **params_no_shuffle)
    input_shape = training_generator.get_input_shape()
    model = get_ResMax(input_shape)
#    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['AUC'])

    EPOCHS = epoch
    lr = LearningRateScheduler(lambda e: sigmoidal_decay(e, end=EPOCHS))
    class_weight = {0: human_weight, 1: 1.} ## human: 0, 1: speaker
    history = model.fit_generator(generator=training_generator, #validation_data=track1_generator,
                              epochs=EPOCHS, class_weight=class_weight, callbacks=[lr], verbose=1)
    eer_val = evalEER(validation_generator, model)
    eer_eval = evalEER(eval_generator, model)
    endtxt1 = str(eer_val)[:6] + '.hdf5'
    savefnm = gen_fname(model_name = mname, params = params , dropout_rate = str(dropout_rate), human_weight = str(human_weight), endtxt = endtxt1 )
    savefnm = 'saved_models/' + savefnm
    model.save(savefnm)
    endtxt1 = str(eer_val)[:6] + '.npy'
    fnm = gen_fname(model_name = mname, params = params , dropout_rate = str(dropout_rate), human_weight = str(human_weight), endtxt = endtxt1)
    fnm = 'saved_results/' + fnm
    sc1 = model.predict(eval_generator)
    np.save(fnm, sc1)
    params_save = params
    params_save['model'] = mname
    params_save['human_weight'] = str(human_weight)
    params_save['dropout_rate'] = str(dropout_rate)
    params_save['eer_val'] = str(eer_val)[:6]
    params_save['eer_eval'] = str(eer_eval)[:6]
    params_save['saved_model'] = savefnm
    tnow = datetime.datetime.now()
    params_save['tnow'] = str(tnow)
    print(params_save)
    fnm = 'res_fa/rec'+str(tnow) + '.pickle'
    with open(fnm, 'wb') as f:
        pickle.dump(params_save, f, pickle.HIGHEST_PROTOCOL)

##  'ffmfa': [[3, 6], 6, "step", [-4,4] ],

for i in range(3) :
    sr = 16000
    sec = 9.0
    batch_size = 16
    feature = "cqt"
    #hop_length = 128
    #win_length = 512
    #n_bins = 80
    filter_scale = 1
    n_bins = 100
    fmin = 5
    epoch = 70
    beta_param = 0.7
    dropout_rate = 0.5 #np.random.choice([0.4, 0.5, 0.6])
    human_weight = 5.0 #np.random.choice([4.0, 5.0, 6.0])
    
    tmp_string = "tmp"
    params = {'sr': sr,
          'batch_size': batch_size,
          'feature': feature,
          'n_classes': 2,
          'sec': sec,
          'filter_scale': filter_scale,
          'fmin' : fmin,
          'n_bins': int(n_bins),
          'tofile': tmp_string,
          'shuffle': True,
          'beta_param': beta_param,
              'ffmfa': [[3, 6], 6, "step", [-4,4] ],
#              'specmix': False,    
#              'cutout': False,
#              'cutmix': False,
          'data_dir': asv2019
#          'lowpass': lowpass,
#          'highpass': highpass,
          #          'ranfilter' : [10,11,12,13,14,15]
#          'ranfilter2' : ranfilter2 
          #           'dropblock' : [30, 100]
          #'device' : device
    }
    params_no_shuffle = {'sr': sr,
                     'batch_size': batch_size,
                     'feature': feature,
                     'n_classes': 2,
                     'sec': sec,
                     'filter_scale': filter_scale,
                     'fmin' : fmin,
                     'n_bins': int(n_bins),
                     'tofile': tmp_string,
                     'shuffle': False,
                    'data_dir': asv2019
                     #'device': device
    }
    training_generator = DataGenerator(dl.train, dl.labels, **params)
    validation_generator = DataGenerator(dl.dev, dl.labels, **params_no_shuffle)
    eval_generator = DataGenerator(dl.eval, dl.labels, **params_no_shuffle)
    input_shape = training_generator.get_input_shape()
    model = get_ResMax(input_shape)
#    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['AUC'])

    EPOCHS = epoch
    lr = LearningRateScheduler(lambda e: sigmoidal_decay(e, end=EPOCHS))
    class_weight = {0: human_weight, 1: 1.} ## human: 0, 1: speaker
    history = model.fit_generator(generator=training_generator, #validation_data=track1_generator,
                              epochs=EPOCHS, class_weight=class_weight, callbacks=[lr], verbose=1)
    eer_val = evalEER(validation_generator, model)
    eer_eval = evalEER(eval_generator, model)
    endtxt1 = str(eer_val)[:6] + '.hdf5'
    savefnm = gen_fname(model_name = mname, params = params , dropout_rate = str(dropout_rate), human_weight = str(human_weight), endtxt = endtxt1 )
    savefnm = 'saved_models/' + savefnm
    model.save(savefnm)
    endtxt1 = str(eer_val)[:6] + '.npy'
    fnm = gen_fname(model_name = mname, params = params , dropout_rate = str(dropout_rate), human_weight = str(human_weight), endtxt = endtxt1)
    fnm = 'saved_results/' + fnm
    sc1 = model.predict(eval_generator)
    np.save(fnm, sc1)
    params_save = params
    params_save['model'] = mname
    params_save['human_weight'] = str(human_weight)
    params_save['dropout_rate'] = str(dropout_rate)
    params_save['eer_val'] = str(eer_val)[:6]
    params_save['eer_eval'] = str(eer_eval)[:6]
    params_save['saved_model'] = savefnm
    tnow = datetime.datetime.now()
    params_save['tnow'] = str(tnow)
    print(params_save)
    fnm = 'res_fa/rec'+str(tnow) + '.pickle'
    with open(fnm, 'wb') as f:
        pickle.dump(params_save, f, pickle.HIGHEST_PROTOCOL)

##  'ffmfa': [[3, 6], 6, "linear", [-4,4] ],

for i in range(3) :
    sr = 16000
    sec = 9.0
    batch_size = 16
    feature = "cqt"
    #hop_length = 128
    #win_length = 512
    #n_bins = 80
    filter_scale = 1
    n_bins = 100
    fmin = 5
    epoch = 70
    beta_param = 0.7
    dropout_rate = 0.5 #np.random.choice([0.4, 0.5, 0.6])
    human_weight = 5.0 #np.random.choice([4.0, 5.0, 6.0])
    
    tmp_string = "tmp"
    params = {'sr': sr,
          'batch_size': batch_size,
          'feature': feature,
          'n_classes': 2,
          'sec': sec,
          'filter_scale': filter_scale,
          'fmin' : fmin,
          'n_bins': int(n_bins),
          'tofile': tmp_string,
          'shuffle': True,
          'beta_param': beta_param,
              'ffmfa': [[3, 6], 6, "linear", [-4,4] ],
#              'specmix': False,    
#              'cutout': False,
#              'cutmix': False,
          'data_dir': asv2019
#          'lowpass': lowpass,
#          'highpass': highpass,
          #          'ranfilter' : [10,11,12,13,14,15]
#          'ranfilter2' : ranfilter2 
          #           'dropblock' : [30, 100]
          #'device' : device
    }
    params_no_shuffle = {'sr': sr,
                     'batch_size': batch_size,
                     'feature': feature,
                     'n_classes': 2,
                     'sec': sec,
                     'filter_scale': filter_scale,
                     'fmin' : fmin,
                     'n_bins': int(n_bins),
                     'tofile': tmp_string,
                     'shuffle': False,
                    'data_dir': asv2019
                     #'device': device
    }
    training_generator = DataGenerator(dl.train, dl.labels, **params)
    validation_generator = DataGenerator(dl.dev, dl.labels, **params_no_shuffle)
    eval_generator = DataGenerator(dl.eval, dl.labels, **params_no_shuffle)
    input_shape = training_generator.get_input_shape()
    model = get_ResMax(input_shape)
#    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['AUC'])

    EPOCHS = epoch
    lr = LearningRateScheduler(lambda e: sigmoidal_decay(e, end=EPOCHS))
    class_weight = {0: human_weight, 1: 1.} ## human: 0, 1: speaker
    history = model.fit_generator(generator=training_generator, #validation_data=track1_generator,
                              epochs=EPOCHS, class_weight=class_weight, callbacks=[lr], verbose=1)
    eer_val = evalEER(validation_generator, model)
    eer_eval = evalEER(eval_generator, model)
    endtxt1 = str(eer_val)[:6] + '.hdf5'
    savefnm = gen_fname(model_name = mname, params = params , dropout_rate = str(dropout_rate), human_weight = str(human_weight), endtxt = endtxt1 )
    savefnm = 'saved_models/' + savefnm
    model.save(savefnm)
    endtxt1 = str(eer_val)[:6] + '.npy'
    fnm = gen_fname(model_name = mname, params = params , dropout_rate = str(dropout_rate), human_weight = str(human_weight), endtxt = endtxt1)
    fnm = 'saved_results/' + fnm
    sc1 = model.predict(eval_generator)
    np.save(fnm, sc1)
    params_save = params
    params_save['model'] = mname
    params_save['human_weight'] = str(human_weight)
    params_save['dropout_rate'] = str(dropout_rate)
    params_save['eer_val'] = str(eer_val)[:6]
    params_save['eer_eval'] = str(eer_eval)[:6]
    params_save['saved_model'] = savefnm
    tnow = datetime.datetime.now()
    params_save['tnow'] = str(tnow)
    print(params_save)
    fnm = 'res_fa/rec'+str(tnow) + '.pickle'
    with open(fnm, 'wb') as f:
        pickle.dump(params_save, f, pickle.HIGHEST_PROTOCOL)



##  'ffmfa': [[3, 6], 10, "step", [-4,4] ],

for i in range(3) :
    sr = 16000
    sec = 9.0
    batch_size = 16
    feature = "cqt"
    #hop_length = 128
    #win_length = 512
    #n_bins = 80
    filter_scale = 1
    n_bins = 100
    fmin = 5
    epoch = 70
    beta_param = 0.7
    dropout_rate = 0.5 #np.random.choice([0.4, 0.5, 0.6])
    human_weight = 5.0 #np.random.choice([4.0, 5.0, 6.0])
    
    tmp_string = "tmp"
    params = {'sr': sr,
          'batch_size': batch_size,
          'feature': feature,
          'n_classes': 2,
          'sec': sec,
          'filter_scale': filter_scale,
          'fmin' : fmin,
          'n_bins': int(n_bins),
          'tofile': tmp_string,
          'shuffle': True,
          'beta_param': beta_param,
              'ffmfa': [[3, 6], 10, "step", [-4,4] ],
#              'specmix': False,    
#              'cutout': False,
#              'cutmix': False,
          'data_dir': asv2019
#          'lowpass': lowpass,
#          'highpass': highpass,
          #          'ranfilter' : [10,11,12,13,14,15]
#          'ranfilter2' : ranfilter2 
          #           'dropblock' : [30, 100]
          #'device' : device
    }
    params_no_shuffle = {'sr': sr,
                     'batch_size': batch_size,
                     'feature': feature,
                     'n_classes': 2,
                     'sec': sec,
                     'filter_scale': filter_scale,
                     'fmin' : fmin,
                     'n_bins': int(n_bins),
                     'tofile': tmp_string,
                     'shuffle': False,
                    'data_dir': asv2019
                     #'device': device
    }
    training_generator = DataGenerator(dl.train, dl.labels, **params)
    validation_generator = DataGenerator(dl.dev, dl.labels, **params_no_shuffle)
    eval_generator = DataGenerator(dl.eval, dl.labels, **params_no_shuffle)
    input_shape = training_generator.get_input_shape()
    model = get_ResMax(input_shape)
#    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['AUC'])

    EPOCHS = epoch
    lr = LearningRateScheduler(lambda e: sigmoidal_decay(e, end=EPOCHS))
    class_weight = {0: human_weight, 1: 1.} ## human: 0, 1: speaker
    history = model.fit_generator(generator=training_generator, #validation_data=track1_generator,
                              epochs=EPOCHS, class_weight=class_weight, callbacks=[lr], verbose=1)
    eer_val = evalEER(validation_generator, model)
    eer_eval = evalEER(eval_generator, model)
    endtxt1 = str(eer_val)[:6] + '.hdf5'
    savefnm = gen_fname(model_name = mname, params = params , dropout_rate = str(dropout_rate), human_weight = str(human_weight), endtxt = endtxt1 )
    savefnm = 'saved_models/' + savefnm
    model.save(savefnm)
    endtxt1 = str(eer_val)[:6] + '.npy'
    fnm = gen_fname(model_name = mname, params = params , dropout_rate = str(dropout_rate), human_weight = str(human_weight), endtxt = endtxt1)
    fnm = 'saved_results/' + fnm
    sc1 = model.predict(eval_generator)
    np.save(fnm, sc1)
    params_save = params
    params_save['model'] = mname
    params_save['human_weight'] = str(human_weight)
    params_save['dropout_rate'] = str(dropout_rate)
    params_save['eer_val'] = str(eer_val)[:6]
    params_save['eer_eval'] = str(eer_eval)[:6]
    params_save['saved_model'] = savefnm
    tnow = datetime.datetime.now()
    params_save['tnow'] = str(tnow)
    print(params_save)
    fnm = 'res_fa/rec'+str(tnow) + '.pickle'
    with open(fnm, 'wb') as f:
        pickle.dump(params_save, f, pickle.HIGHEST_PROTOCOL)

##  'ffmfa': [[3, 6], 6, "linear", [-4,4] ],

for i in range(3) :
    sr = 16000
    sec = 9.0
    batch_size = 16
    feature = "cqt"
    #hop_length = 128
    #win_length = 512
    #n_bins = 80
    filter_scale = 1
    n_bins = 100
    fmin = 5
    epoch = 70
    beta_param = 0.7
    dropout_rate = 0.5 #np.random.choice([0.4, 0.5, 0.6])
    human_weight = 5.0 #np.random.choice([4.0, 5.0, 6.0])
    
    tmp_string = "tmp"
    params = {'sr': sr,
          'batch_size': batch_size,
          'feature': feature,
          'n_classes': 2,
          'sec': sec,
          'filter_scale': filter_scale,
          'fmin' : fmin,
          'n_bins': int(n_bins),
          'tofile': tmp_string,
          'shuffle': True,
          'beta_param': beta_param,
              'ffmfa': [[3, 6], 6, "linear", [-4,4] ],
#              'specmix': False,    
#              'cutout': False,
#              'cutmix': False,
          'data_dir': asv2019
#          'lowpass': lowpass,
#          'highpass': highpass,
          #          'ranfilter' : [10,11,12,13,14,15]
#          'ranfilter2' : ranfilter2 
          #           'dropblock' : [30, 100]
          #'device' : device
    }
    params_no_shuffle = {'sr': sr,
                     'batch_size': batch_size,
                     'feature': feature,
                     'n_classes': 2,
                     'sec': sec,
                     'filter_scale': filter_scale,
                     'fmin' : fmin,
                     'n_bins': int(n_bins),
                     'tofile': tmp_string,
                     'shuffle': False,
                    'data_dir': asv2019
                     #'device': device
    }
    training_generator = DataGenerator(dl.train, dl.labels, **params)
    validation_generator = DataGenerator(dl.dev, dl.labels, **params_no_shuffle)
    eval_generator = DataGenerator(dl.eval, dl.labels, **params_no_shuffle)
    input_shape = training_generator.get_input_shape()
    model = get_ResMax(input_shape)
#    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['AUC'])

    EPOCHS = epoch
    lr = LearningRateScheduler(lambda e: sigmoidal_decay(e, end=EPOCHS))
    class_weight = {0: human_weight, 1: 1.} ## human: 0, 1: speaker
    history = model.fit_generator(generator=training_generator, #validation_data=track1_generator,
                              epochs=EPOCHS, class_weight=class_weight, callbacks=[lr], verbose=1)
    eer_val = evalEER(validation_generator, model)
    eer_eval = evalEER(eval_generator, model)
    endtxt1 = str(eer_val)[:6] + '.hdf5'
    savefnm = gen_fname(model_name = mname, params = params , dropout_rate = str(dropout_rate), human_weight = str(human_weight), endtxt = endtxt1 )
    savefnm = 'saved_models/' + savefnm
    model.save(savefnm)
    endtxt1 = str(eer_val)[:6] + '.npy'
    fnm = gen_fname(model_name = mname, params = params , dropout_rate = str(dropout_rate), human_weight = str(human_weight), endtxt = endtxt1)
    fnm = 'saved_results/' + fnm
    sc1 = model.predict(eval_generator)
    np.save(fnm, sc1)
    params_save = params
    params_save['model'] = mname
    params_save['human_weight'] = str(human_weight)
    params_save['dropout_rate'] = str(dropout_rate)
    params_save['eer_val'] = str(eer_val)[:6]
    params_save['eer_eval'] = str(eer_eval)[:6]
    params_save['saved_model'] = savefnm
    tnow = datetime.datetime.now()
    params_save['tnow'] = str(tnow)
    print(params_save)
    fnm = 'res_fa/rec'+str(tnow) + '.pickle'
    with open(fnm, 'wb') as f:
        pickle.dump(params_save, f, pickle.HIGHEST_PROTOCOL)


##  'ffmfa': [[3, 6], 10, "step", [-4,4] ],

for i in range(3) :
    sr = 16000
    sec = 9.0
    batch_size = 16
    feature = "cqt"
    #hop_length = 128
    #win_length = 512
    #n_bins = 80
    filter_scale = 1
    n_bins = 100
    fmin = 5
    epoch = 70
    beta_param = 0.7
    dropout_rate = 0.5 #np.random.choice([0.4, 0.5, 0.6])
    human_weight = 5.0 #np.random.choice([4.0, 5.0, 6.0])
    
    tmp_string = "tmp"
    params = {'sr': sr,
          'batch_size': batch_size,
          'feature': feature,
          'n_classes': 2,
          'sec': sec,
          'filter_scale': filter_scale,
          'fmin' : fmin,
          'n_bins': int(n_bins),
          'tofile': tmp_string,
          'shuffle': True,
          'beta_param': beta_param,
              'ffmfa': [[3, 6], 10, "step", [-4,4] ],
#              'specmix': False,    
#              'cutout': False,
#              'cutmix': False,
          'data_dir': asv2019
#          'lowpass': lowpass,
#          'highpass': highpass,
          #          'ranfilter' : [10,11,12,13,14,15]
#          'ranfilter2' : ranfilter2 
          #           'dropblock' : [30, 100]
          #'device' : device
    }
    params_no_shuffle = {'sr': sr,
                     'batch_size': batch_size,
                     'feature': feature,
                     'n_classes': 2,
                     'sec': sec,
                     'filter_scale': filter_scale,
                     'fmin' : fmin,
                     'n_bins': int(n_bins),
                     'tofile': tmp_string,
                     'shuffle': False,
                    'data_dir': asv2019
                     #'device': device
    }
    training_generator = DataGenerator(dl.train, dl.labels, **params)
    validation_generator = DataGenerator(dl.dev, dl.labels, **params_no_shuffle)
    eval_generator = DataGenerator(dl.eval, dl.labels, **params_no_shuffle)
    input_shape = training_generator.get_input_shape()
    model = get_ResMax(input_shape)
#    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['AUC'])

    EPOCHS = epoch
    lr = LearningRateScheduler(lambda e: sigmoidal_decay(e, end=EPOCHS))
    class_weight = {0: human_weight, 1: 1.} ## human: 0, 1: speaker
    history = model.fit_generator(generator=training_generator, #validation_data=track1_generator,
                              epochs=EPOCHS, class_weight=class_weight, callbacks=[lr], verbose=1)
    eer_val = evalEER(validation_generator, model)
    eer_eval = evalEER(eval_generator, model)
    endtxt1 = str(eer_val)[:6] + '.hdf5'
    savefnm = gen_fname(model_name = mname, params = params , dropout_rate = str(dropout_rate), human_weight = str(human_weight), endtxt = endtxt1 )
    savefnm = 'saved_models/' + savefnm
    model.save(savefnm)
    endtxt1 = str(eer_val)[:6] + '.npy'
    fnm = gen_fname(model_name = mname, params = params , dropout_rate = str(dropout_rate), human_weight = str(human_weight), endtxt = endtxt1)
    fnm = 'saved_results/' + fnm
    sc1 = model.predict(eval_generator)
    np.save(fnm, sc1)
    params_save = params
    params_save['model'] = mname
    params_save['human_weight'] = str(human_weight)
    params_save['dropout_rate'] = str(dropout_rate)
    params_save['eer_val'] = str(eer_val)[:6]
    params_save['eer_eval'] = str(eer_eval)[:6]
    params_save['saved_model'] = savefnm
    tnow = datetime.datetime.now()
    params_save['tnow'] = str(tnow)
    print(params_save)
    fnm = 'res_fa/rec'+str(tnow) + '.pickle'
    with open(fnm, 'wb') as f:
        pickle.dump(params_save, f, pickle.HIGHEST_PROTOCOL)

##  'ffmfa': [[3, 6], 10, "linear", [-4,4] ],

for i in range(3) :
    sr = 16000
    sec = 9.0
    batch_size = 16
    feature = "cqt"
    #hop_length = 128
    #win_length = 512
    #n_bins = 80
    filter_scale = 1
    n_bins = 100
    fmin = 5
    epoch = 70
    beta_param = 0.7
    dropout_rate = 0.5 #np.random.choice([0.4, 0.5, 0.6])
    human_weight = 5.0 #np.random.choice([4.0, 5.0, 6.0])
    
    tmp_string = "tmp"
    params = {'sr': sr,
          'batch_size': batch_size,
          'feature': feature,
          'n_classes': 2,
          'sec': sec,
          'filter_scale': filter_scale,
          'fmin' : fmin,
          'n_bins': int(n_bins),
          'tofile': tmp_string,
          'shuffle': True,
          'beta_param': beta_param,
              'ffmfa': [[3, 6], 10, "linear", [-4,4] ],
#              'specmix': False,    
#              'cutout': False,
#              'cutmix': False,
          'data_dir': asv2019
#          'lowpass': lowpass,
#          'highpass': highpass,
          #          'ranfilter' : [10,11,12,13,14,15]
#          'ranfilter2' : ranfilter2 
          #           'dropblock' : [30, 100]
          #'device' : device
    }
    params_no_shuffle = {'sr': sr,
                     'batch_size': batch_size,
                     'feature': feature,
                     'n_classes': 2,
                     'sec': sec,
                     'filter_scale': filter_scale,
                     'fmin' : fmin,
                     'n_bins': int(n_bins),
                     'tofile': tmp_string,
                     'shuffle': False,
                    'data_dir': asv2019
                     #'device': device
    }
    training_generator = DataGenerator(dl.train, dl.labels, **params)
    validation_generator = DataGenerator(dl.dev, dl.labels, **params_no_shuffle)
    eval_generator = DataGenerator(dl.eval, dl.labels, **params_no_shuffle)
    input_shape = training_generator.get_input_shape()
    model = get_ResMax(input_shape)
#    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['AUC'])

    EPOCHS = epoch
    lr = LearningRateScheduler(lambda e: sigmoidal_decay(e, end=EPOCHS))
    class_weight = {0: human_weight, 1: 1.} ## human: 0, 1: speaker
    history = model.fit_generator(generator=training_generator, #validation_data=track1_generator,
                              epochs=EPOCHS, class_weight=class_weight, callbacks=[lr], verbose=1)
    eer_val = evalEER(validation_generator, model)
    eer_eval = evalEER(eval_generator, model)
    endtxt1 = str(eer_val)[:6] + '.hdf5'
    savefnm = gen_fname(model_name = mname, params = params , dropout_rate = str(dropout_rate), human_weight = str(human_weight), endtxt = endtxt1 )
    savefnm = 'saved_models/' + savefnm
    model.save(savefnm)
    endtxt1 = str(eer_val)[:6] + '.npy'
    fnm = gen_fname(model_name = mname, params = params , dropout_rate = str(dropout_rate), human_weight = str(human_weight), endtxt = endtxt1)
    fnm = 'saved_results/' + fnm
    sc1 = model.predict(eval_generator)
    np.save(fnm, sc1)
    params_save = params
    params_save['model'] = mname
    params_save['human_weight'] = str(human_weight)
    params_save['dropout_rate'] = str(dropout_rate)
    params_save['eer_val'] = str(eer_val)[:6]
    params_save['eer_eval'] = str(eer_eval)[:6]
    params_save['saved_model'] = savefnm
    tnow = datetime.datetime.now()
    params_save['tnow'] = str(tnow)
    print(params_save)
    fnm = 'res_fa/rec'+str(tnow) + '.pickle'
    with open(fnm, 'wb') as f:
        pickle.dump(params_save, f, pickle.HIGHEST_PROTOCOL)
        
