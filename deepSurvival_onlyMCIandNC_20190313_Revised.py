import sys, os, datetime, h5py
import numpy as np
from scipy import io
from matplotlib import pyplot
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Input,InputLayer, Dense,  Dropout, Activation, Concatenate, Lambda
#from tensorflow.python.keras.summary import merge
from tensorflow.keras.utils import plot_model, multi_gpu_model
from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler, ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
from functools import partial, update_wrapper
from tensorflow.keras import optimizers

from sklearn.preprocessing import StandardScaler

from lifelines.utils import concordance_index

np.random.seed(20180224)
np.random.seed(np.random.randint(100000))

datadir = r"/media/dl-box/Elements/DeepLearning/GLM246/"

isAAL = 0
combined = 0
withAge = 0
withMMSE = 0;

CVs = 10
Itr = 1 # max 5
DROPOUT_RATIO = 0.5
NB_EPOCH = 100
BATCH_SIZE = 128
N_GPUS = 4
#MRI = 2 # 1:1.5T 2: 3.0T

for ii in range(Itr):
    os.chdir( datadir )
    if isAAL == 0:
        data = io.loadmat("GMV.mat")
        gmv = np.array( data["GMV"], dtype = 'float32' )
        data = io.loadmat("GMV_Shimane.mat")
        gmv_shimaneH = np.array( data["GMV"], dtype = 'float32' )
        data = io.loadmat("GMV_Dock.mat")
        gmv_dock = np.array( data["GMV"], dtype = 'float32' )
    else:
        data = io.loadmat("GMV_AAL.mat")
        gmv = np.array( data["GMV"], dtype = 'float32' )
        data = io.loadmat("GMV_Shimane_AAL.mat")
        gmv_shimaneH = np.array( data["GMV"], dtype = 'float32' )
        data = io.loadmat("GMV_Dock_AAL.mat")
        gmv_dock = np.array( data["GMV"], dtype = 'float32' )
    if combined == 1:
        data = io.loadmat("GMV.mat")
        gmv1 = np.array( data["GMV"], dtype = 'float32' )
        data = io.loadmat("GMV_AAL.mat")
        gmv2 = np.array( data["GMV"], dtype = 'float32' )
        gmv = np.concatenate([gmv1, gmv2], axis=1)
        data = io.loadmat("GMV_Shimane.mat")
        gmv_shimaneH1 = np.array( data["GMV"], dtype = 'float32' )
        data = io.loadmat("GMV_Dock.mat")
        gmv_dock1 = np.array( data["GMV"], dtype = 'float32' )
        data = io.loadmat("GMV_Shimane_AAL.mat")
        gmv_shimaneH2 = np.array( data["GMV"], dtype = 'float32' )
        data = io.loadmat("GMV_Dock_AAL.mat")
        gmv_dock2 = np.array( data["GMV"], dtype = 'float32' )
        gmv_shimaneH = np.concatenate([gmv_shimaneH1, gmv_shimaneH2], axis=1)
        gmv_dock = np.concatenate([gmv_dock1, gmv_dock2], axis=1)


    data  = io.loadmat("Demo_MMSE_corrected.mat" )
    demo = np.array( data["Demo"], dtype = 'float32'  )   # database group mri subjectID age sex convert interval MMSE
    data  = io.loadmat("Demo_Shimane.mat" )
    demo_shimaneH = np.array( data["Demo_Shimane"], dtype = 'float32'  )   # age sex convert interval
    data  = io.loadmat("Demo_Dock.mat" )
    demo_dock = np.array( data["Demo"], dtype = 'float32'  )   # age sex convert interval mmse

    index1 = (demo_shimaneH[:,2]==0)*(demo_shimaneH[:,0]>100)
    index2 = (demo_shimaneH[:,0]<=63)
    nn_shimaneH = np.sum(index1==0)
    nn_dock = np.sum(index2==0)
    gmv_shimane = np.concatenate([gmv_shimaneH[index1==0,:], gmv_dock],axis=0)
    demo_shimane = np.concatenate([demo_shimaneH[index1==0,:], demo_dock],axis=0)
    nn_shimane = demo_shimane.shape[0]
    print(gmv_shimane.shape)
    print(demo_shimane.shape)
    features_test = gmv_shimane;
    if withAge == 1:
        age = demo_shimane[:,0]/100
        age = age[:,np.newaxis]
        features_test = np.concatenate([features_test, age], axis=1)
    if withMMSE == 1:
        mmse = demo_shimane[:,4]/30
        mmse = mmse[:,np.newaxis]
        features_test = np.concatenate([features_test, mmse], axis=1)

    print(np.mean(demo[ (demo[:,0]==1)*(demo[:,7]>0) ,4]),np.std(demo[ (demo[:,0]==1)*(demo[:,7]>0),4]),np.sum(demo[ (demo[:,0]==1)*(demo[:,7]>0) ,6],0) )
    print(np.mean(demo[ (demo[:,0]==2)*(demo[:,7]>0) ,4]),np.std(demo[ (demo[:,0]==2)*(demo[:,7]>0),4]),np.sum(demo[ (demo[:,0]==2)*(demo[:,7]>0) ,6],0) )
    print(np.mean(demo[ (demo[:,0]==3)*(demo[:,7]>0) ,4]),np.std(demo[ (demo[:,0]==3)*(demo[:,7]>0),4]),np.sum(demo[ (demo[:,0]==3)*(demo[:,7]>0) ,6],0) )
    print('ADNI MCI age')
    print(np.mean(demo[ (demo[:,0]==1)*(demo[:,7]>0) * (demo[:,1]==2),4]),np.std(demo[ (demo[:,0]==1)*(demo[:,7]>0) * (demo[:,1]==2),4]),np.sum(demo[ (demo[:,0]==1)*(demo[:,7]>0) * (demo[:,1]==2),6],0) )
    print('ADNI NC age')
    print(np.mean(demo[ (demo[:,0]==1)*(demo[:,7]>0) * (demo[:,1]==3),4]),np.std(demo[ (demo[:,0]==1)*(demo[:,7]>0) * (demo[:,1]==3),4]),np.sum(demo[ (demo[:,0]==1)*(demo[:,7]>0) * (demo[:,1]==3),6],0))
    print('AIBL MCI age')
    print(np.mean(demo[ (demo[:,0]==2)*(demo[:,7]>0) * (demo[:,1]==2),4]),np.std(demo[ (demo[:,0]==2)*(demo[:,7]>0) * (demo[:,1]==2),4]),np.sum(demo[ (demo[:,0]==2)*(demo[:,7]>0) * (demo[:,1]==2),6],0))
    print('AIBL NC age')
    print(np.mean(demo[ (demo[:,0]==2)*(demo[:,7]>0) * (demo[:,1]==3),4]),np.std(demo[ (demo[:,0]==2)*(demo[:,7]>0) * (demo[:,1]==3),4]),np.sum(demo[ (demo[:,0]==2)*(demo[:,7]>0) * (demo[:,1]==3),6],0))
    print('JADNI MCI age')
    print(np.mean(demo[ (demo[:,0]==3)*(demo[:,7]>0) * (demo[:,1]==2),4]),np.std(demo[ (demo[:,0]==3)*(demo[:,7]>0) * (demo[:,1]==2),4]),np.sum(demo[ (demo[:,0]==3)*(demo[:,7]>0) * (demo[:,1]==2),6],0))
    print('JADNI NC age')
    print(np.mean(demo[ (demo[:,0]==3)*(demo[:,7]>0) * (demo[:,1]==3),4]),np.std(demo[ (demo[:,0]==3)*(demo[:,7]>0) * (demo[:,1]==3),4]),np.sum(demo[ (demo[:,0]==3)*(demo[:,7]>0) * (demo[:,1]==3),6],0))
    print('3DB MCI age')
    print(np.mean(demo[ (demo[:,1]==2)*(demo[:,7]>0),4]),np.std(demo[ (demo[:,1]==2),4]),np.sum(demo[ (demo[:,1]==2),6],0) )
    print('3DB NC age')
    print(np.mean(demo[ (demo[:,1]==3)*(demo[:,7]>0),4]),np.std(demo[ (demo[:,1]==3),4]),np.sum(demo[ (demo[:,1]==3),6],0) )
    print('Shimane MCI age')
    print(np.mean(demo_shimane[:,0],0),np.std(demo_shimane[:,0],0),np.sum(demo_shimane[:,2],0))
    print(np.mean(demo_shimane[0:nn_shimaneH,0],0),np.std(demo_shimane[0:nn_shimaneH,0],0),np.sum(demo_shimane[0:nn_shimaneH,2],0))
    print(np.mean(demo_shimane[nn_shimaneH:nn_shimane,0],0),np.std(demo_shimane[nn_shimaneH:nn_shimane,0],0),np.sum(demo_shimane[nn_shimaneH:nn_shimane,2],0))
    print(np.max(demo_shimane[:,3],0))

    if withMMSE == 1:
        target = (demo[:,1] > 1)*(demo[:,7]>0) * (demo[:,8]>0)
    else:
        target = (demo[:,1] > 1)*(demo[:,7]>0)

    target[gmv[:,0]==0] = 0
    target_ADNI = (demo[target,0] == 1)
    target_AIBL = (demo[target,0] == 2)
    target_JADNI = (demo[target,0] == 3)

    features = gmv[ target, :]
    if withAge == 1:
        age = demo[ target, 4]/100
        age = age[:,np.newaxis]
        features = np.concatenate([features, age], axis=1)
    if withMMSE == 1:
        mmse = demo[ target, 8]/30
        mmse = mmse[:,np.newaxis]
        features = np.concatenate([features, age], axis=1)
    n_features =   features.shape[1]
    e = np.array(demo[ target, 6], dtype = 'int32')
    t = demo[ target, 7]

    nn = features.shape[0]
    training_sample = np.arange(nn)
    np.random.shuffle(training_sample)
    I = np.argsort(training_sample)
    cv = training_sample % CVs
    features = features[training_sample,:]
    e = e[training_sample]
    t = t[training_sample]
    #J = np.int(np.ceil(np.max(t)) + 1)
    J = np.int( np.round( np.max(t) ) + 1 )
    print('time points:',J)
    print('samples: ', nn)

    #T = np.ceil(t).astype('int64')
    T = np.round(t).astype('int64')
    T[T==0] = 1
    E = np.zeros([nn,J])
    mask = np.ones([nn,J]).astype('float32')
    for ii in range(nn):
        if e[ii] == 1:
            E[ii,T[ii]:J] = 1
        if e[ii] == 0:
            mask[ii,T[ii]+1:J] = 0

    mask_shimane = np.ones([nn,J]).astype('float32')
    nn_shimane = features_test.shape[0]
    e_shimane = np.array(demo_shimane[ :, 2], dtype = 'int32')
    E_shimane = np.zeros([nn_shimane,J])
    T_shimane = np.round(demo_shimane[ :, 3]).astype('int64')
    T_shimane[T_shimane==0] = 1
    for ii in range(nn_shimane):
        if e_shimane[ii] == 1:
            E_shimane[ii,T_shimane[ii]:J] = 1
        if e_shimane[ii] == 0:
            mask_shimane[ii,T_shimane[ii]+1:J] = 0

    todaydetail  =    datetime.datetime.today()
    logdir = datadir + "log_DeepSurv_" + todaydetail.strftime("%Y%m%d_%H%M")
    os.makedirs(logdir, exist_ok = True)
    os.chdir(logdir)
    experiment_name = 'deepsurv'

    from tensorflow.python.client import device_lib
    device_lib.list_local_devices()

    with tf.device("/cpu:0"):

        def output_of_lambda(input_shape):
            shape = list(input_shape)
            return (shape[0], J)

        def weibull_cdf(parameters):
            m = parameters[:,0]
            s = tf.maximum( parameters[:,1], 0.001 )
            output_list = []
            for ii in range( J ):
                Time   = tf.constant( ii, dtype="float32")
                e_Time = tf.pow( Time, m )
                s_Time = tf.negative( tf.div( e_Time, s) )
                x = tf.subtract( tf.constant(1, dtype="float32") , tf.exp( s_Time ) ) # F(t) = 1 - exp(-(t-g)^m/s) #ref http://www.mogami.com/notes/weibull.html
                output_list.append ( x )
            return tf.stack(output_list, axis=1)

        def generator_loss(y_true, y_pred, weights):  # y_true's shape=(batch_size, row, col, ch)
            #loss = tf.cumsum( tf.multiply( tf.square( tf.subtract( y_pred, y_true ) ), weights ), axis=1, reverse=True)[:,0]
            log_p = tf.log( tf.add( y_pred,  tf.constant(1.0) ) )
            log_t = tf.log( tf.add( y_true,  tf.constant(1.0) ) )
            loss = tf.cumsum( tf.multiply( tf.square( tf.subtract( log_p, log_t ) ), weights ), axis=1, reverse=True)[:,0]
            return loss

        def wrapped_generator_loss(func, *args, **kwargs):
            partial_generator_loss = partial(generator_loss, *args, **kwargs)
            update_wrapper(partial_generator_loss, generator_loss)
            return partial_generator_loss

        inputs = Input((n_features,), name='inputs')
        x1 = Dense(units=32, activation='relu', name='hidden_layer1')(inputs)
        x1 = Dropout(DROPOUT_RATIO)(x1)
        x2 = Dense(units=32, activation='relu', name='hidden_layer2')(x1)
        x2 = Dropout(DROPOUT_RATIO)(x2)
        x3 = Dense(units=32, activation='relu', name='hidden_layer3')(x2)
        x3 = Dropout(DROPOUT_RATIO)(x3)
        p1 = Dense(units=1, activation='softplus', name='param1_layer')(x3)
        p2 = Dense(units=1, activation='relu', name='param2_layer')(x3)
        parameters = Concatenate(name='params_layer')([p1, p2])
        y_pred = Lambda(weibull_cdf, output_shape=output_of_lambda)(parameters)

        mask_batch = Input((J,), name='mask_bartch')
        L = wrapped_generator_loss(generator_loss, weights=mask_batch)

        model = Model(inputs= [inputs, mask_batch], outputs = y_pred)
        model.summary()

    models = multi_gpu_model(model, gpus=N_GPUS)

    pred_params = np.zeros([nn,2])
    c_index_shimane = np.zeros([J,CVs])
    for num in range(CVs):

        x_train = features[cv != num,:]
        y_train = E[cv != num,:]
        mask_train = mask[cv != num,:]
        x_test = features[cv == num,:]
        y_test = E[cv == num,:]
        mask_test = mask[cv == num,:]

        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        x_val = scaler.transform(features_test)

        todaydetail  =    datetime.datetime.today()
        outputfilename     = 'Training___CV' + str(num) + '_Itr_' + todaydetail.strftime("_%Y%m%d_%H%M") + '.csv'
        weightfilename     = 'WeightBest_CV' + str(num) + '_Itr_' + todaydetail.strftime("_%Y%m%d_%H%M") + '.h5'

        checkpointer = ModelCheckpoint(filepath=weightfilename, monitor='loss', verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss',patience=10,verbose=1)
        callbacks = []
        callbacks.append(early_stopping)
        callbacks.append(CSVLogger(outputfilename))
        #callbacks.append(checkpointer)

        adm = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        models.compile(optimizer=adm, loss=L)
        #models.compile(optimizer='Adam', loss='mean_squared_error', metrics=["accuracy"])
        session = K.get_session()
        if num==0:
            for layer in model.layers:
                if hasattr(layer, 'kernel_initializer'):
                    layer.kernel.initializer.run(session=session)
                    model.save_weights('InitialWeights.h5')
                    print('save initial weights')
        elif num>0:
            model.load_weights('InitialWeights.h5')
            print('load inital weights')

        #models.fit([x_train, mask_train], y_train, batch_size=BATCH_SIZE, epochs = NB_EPOCH, callbacks=callbacks, verbose=1)
        # version of first subimission
        models.fit([x_train, mask_train], y_train, batch_size=BATCH_SIZE, epochs = NB_EPOCH, callbacks=callbacks, verbose=2, validation_data = ([x_test, mask_test], y_test))
        #models.fit([x_train, mask_train], y_train, batch_size=BATCH_SIZE, epochs = NB_EPOCH, callbacks=callbacks, verbose=2, validation_split=0.1)

        #intermediate_model = Model(inputs=model.input, outputs=model.get_layer('params_layer').output)

        #models.load_weights(weightfilename)
        #models.compile(optimizer='Adam', loss=L)
        prob = models.predict([x_test, mask_test], batch_size=BATCH_SIZE, verbose=1)
        intermediate_model = Model(inputs=model.input, outputs=model.get_layer('params_layer').output)
        intermediate_output = intermediate_model.predict([x_test, mask_test])
        pred_params[cv == num,:] = intermediate_output

        pred_params_shimane = np.zeros([nn_shimane,2])
        pred_params_shimane = intermediate_model.predict([x_val, mask_shimane])
        pred_prob_shimane = np.zeros([nn_shimane,J])
        for tt in range(J):
            pred_prob_shimane[:,tt] = 1 - np.exp( - tt ** pred_params_shimane[:,0]  / pred_params_shimane[:,1] );
        todaydetail  =    datetime.datetime.today()
        predictionfilename = 'Para_' + str(num+1) + todaydetail.strftime("_%Y%m%d_%H%M") + '.csv'
        prediction = np.c_[pred_params_shimane, pred_prob_shimane]
        np.savetxt(predictionfilename, prediction, delimiter=',')

        for num2 in range(J - 1) :
            c_index_shimane[num2+1,num] = concordance_index(T_shimane,1/pred_prob_shimane[:,num2+1], e_shimane)
        print( c_index_shimane )

    pred_prob = np.zeros([nn,J])
    for tt in range(J):
        pred_prob[:,tt] = 1 - np.exp( - tt ** pred_params[:,0]  / pred_params[:,1] );


    todaydetail  =    datetime.datetime.today()
    predictionfilename = 'Prediction_CV' + str(CVs) + todaydetail.strftime("_%Y%m%d_%H%M") + '.csv'
    prediction = np.c_[training_sample[I], demo[target,0:6], e[I], t[I], pred_params[I,:], pred_prob[I,:], I]
    np.savetxt(predictionfilename, prediction, delimiter=',')

    c_index = np.zeros([J])
    for num in range(J - 1) :
        c_index[num+1] = concordance_index(T,1/pred_prob[:,num+1], e)
    print( c_index )

    c_index_ADNI = np.zeros([J])
    for num in range(J - 1) :
        c_index_ADNI[num+1] = concordance_index(T[target_ADNI],1/pred_prob[target_ADNI,num+1], e[target_ADNI])
    print( c_index_ADNI )

    c_index_AIBL = np.zeros([J])
    for num in range(J - 1) :
        c_index_AIBL[num+1] = concordance_index(T[target_AIBL],1/pred_prob[target_AIBL,num+1], e[target_AIBL])
    print( c_index_AIBL )

    c_index_JADNI = np.zeros([J])
    for num in range(J - 1) :
        c_index_JADNI[num+1] = concordance_index(T[target_JADNI],1/pred_prob[target_JADNI,num+1], e[target_JADNI])
    print( c_index_JADNI )

    cindexfilename = 'C_index_' + todaydetail.strftime("_%Y%m%d_%H%M") + '.txt'
    np.savetxt(cindexfilename, (c_index,c_index_ADNI, c_index_AIBL, c_index_JADNI))
    cindexfilename2 = 'C_index_shiamane_' + todaydetail.strftime("_%Y%m%d_%H%M") + '.txt'
    np.savetxt(cindexfilename2, c_index_shimane)

    json_string = models.to_json()
    modeltxtfilename   = 'Modeltxt_' + todaydetail.strftime("_%Y%m%d_%H%M") + '.txt'
    f = open(modeltxtfilename,'w')
    f.write(json_string)
    f.close()
