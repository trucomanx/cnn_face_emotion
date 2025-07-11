#!/usr/bin/python

import os
import sys
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from tensorflow import Tensor


def dense_block1( x:           Tensor,
                    filters_output, 
                    filters_hidden: int = 16, 
                    kernel_size:    int = 7,
                    alpha=0.05) -> Tensor:
    
    ##
    y = tf.keras.layers.Conv2D(kernel_size = kernel_size,
                               strides     = 1,
                               filters     = filters_hidden,
                               padding     = "same")(x);
    y = tf.keras.layers.LeakyReLU(alpha=alpha)(y);
    
    ##
    y = tf.keras.layers.Conv2D(kernel_size = kernel_size,
                               strides     = 1,
                               filters     = filters_hidden,
                               padding     = "same")(x);
    y = tf.keras.layers.LeakyReLU(alpha=alpha)(y);

    ##
    y = tf.keras.layers.Conv2D(kernel_size = kernel_size,
                               strides     = 1,
                               filters     = filters_output,
                               padding     = "same")(x);
    y = tf.keras.layers.LeakyReLU(alpha=alpha)(y);
    
    out = tf.keras.layers.Concatenate(axis=-1)([x, y])
    
    out = tf.keras.layers.BatchNormalization()(out);
    
    return out;

def create_model_dense1(enable_summary=True,Nout=14):
    target_size=(256,256);
    n_filters=7;
    nh_filters=7;
    
    
    inputs = tf.keras.layers.Input(shape=(target_size[0],target_size[1], 3));
    
    ########
    t = tf.keras.layers.Conv2D(kernel_size = 5,
                               strides     = 1,
                               filters     = n_filters,
                               activation  = "relu",
                               padding     = "same")(inputs);
    ########
    
    t = dense_block1( t, filters_output = n_filters, filters_hidden = nh_filters, kernel_size = 5);
    
    t = dense_block1( t, filters_output = n_filters, filters_hidden = nh_filters, kernel_size = 5);
    
    t = tf.keras.layers.MaxPooling2D(pool_size=2)(t);
    
    ########
    
    t = dense_block1( t, filters_output = n_filters, filters_hidden = nh_filters, kernel_size = 5);
    
    t = dense_block1( t, filters_output = n_filters, filters_hidden = nh_filters, kernel_size = 5);
    
    t = tf.keras.layers.MaxPooling2D(pool_size=2)(t);
    
    ########
    
    t = dense_block1( t, filters_output = n_filters, filters_hidden = nh_filters, kernel_size = 5);
    
    t = dense_block1( t, filters_output = n_filters, filters_hidden = nh_filters, kernel_size = 5);
    
    t = tf.keras.layers.MaxPooling2D(pool_size=2)(t);
    
    ########
    
    t = dense_block1( t, filters_output = n_filters, filters_hidden = nh_filters, kernel_size = 3);
    
    t = dense_block1( t, filters_output = n_filters, filters_hidden = nh_filters, kernel_size = 3);
    
    t = tf.keras.layers.AveragePooling2D(pool_size=2)(t);
    
    ########
    
    t = dense_block1( t, filters_output = n_filters, filters_hidden = nh_filters, kernel_size = 3);
    
    t = dense_block1( t, filters_output = n_filters, filters_hidden = nh_filters, kernel_size = 3);
    
    t = tf.keras.layers.AveragePooling2D(pool_size=2)(t);
    
    ########
    
    t = dense_block1( t, filters_output = n_filters, filters_hidden = nh_filters, kernel_size = 3);
    
    t = dense_block1( t, filters_output = n_filters, filters_hidden = nh_filters, kernel_size = 3);
    
    t = tf.keras.layers.AveragePooling2D(pool_size=2)(t);
    
    ########
    
    t = tf.keras.layers.Flatten()(t);
    
    outputs = tf.keras.layers.Dense(Nout, activation='tanh')(t);
    
    model = tf.keras.models.Model(inputs, outputs);
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy']);
    
    if enable_summary:
        model.summary();
    
    return model, target_size;

def residual_block1( x:           Tensor,
                    filters_output, 
                    filters_hidden: int = 16, 
                    kernel_size:    int = 7,
                    alpha=0.05) -> Tensor:
    
    ##
    y = tf.keras.layers.Conv2D(kernel_size = kernel_size,
                               strides     = 1,
                               filters     = filters_hidden,
                               padding     = "same")(x);
    y = tf.keras.layers.LeakyReLU(alpha=alpha)(y);
    
    ##
    y = tf.keras.layers.Conv2D(kernel_size = kernel_size,
                               strides     = 1,
                               filters     = filters_hidden,
                               padding     = "same")(x);
    y = tf.keras.layers.LeakyReLU(alpha=alpha)(y);

    ##
    y = tf.keras.layers.Conv2D(kernel_size = kernel_size,
                               strides     = 1,
                               filters     = filters_output,
                               padding     = "same")(x);
    
    out = tf.keras.layers.Add()([x, y]);
    
    out = tf.keras.layers.LeakyReLU(alpha=alpha)(out);
    
    out = tf.keras.layers.BatchNormalization()(out);
    
    return out;
    
def create_model_residual1(enable_summary=True,Nout=14):
    target_size=(256,256);
    n_filters=14;
    nh_filters=7;
    
    
    inputs = tf.keras.layers.Input(shape=(target_size[0],target_size[1], 3));
    
    ########
    t = tf.keras.layers.Conv2D(kernel_size = 7,
                               strides     = 1,
                               filters     = n_filters,
                               activation  = "relu",
                               padding     = "same")(inputs);
    ########
    
    t = residual_block1( t, filters_output = n_filters, filters_hidden = nh_filters, kernel_size = 7);
    
    t = residual_block1( t, filters_output = n_filters, filters_hidden = nh_filters, kernel_size = 7);
    
    t = tf.keras.layers.MaxPooling2D(pool_size=2)(t);
    
    ########
    
    t = residual_block1( t, filters_output = n_filters, filters_hidden = nh_filters, kernel_size = 5);
    
    t = residual_block1( t, filters_output = n_filters, filters_hidden = nh_filters, kernel_size = 5);
    
    t = tf.keras.layers.MaxPooling2D(pool_size=2)(t);
    
    ########
    
    t = residual_block1( t, filters_output = n_filters, filters_hidden = nh_filters, kernel_size = 5);
    
    t = residual_block1( t, filters_output = n_filters, filters_hidden = nh_filters, kernel_size = 5);
    
    t = tf.keras.layers.MaxPooling2D(pool_size=2)(t);
    
    ########
    
    t = residual_block1( t, filters_output = n_filters, filters_hidden = nh_filters, kernel_size = 3);
    
    t = residual_block1( t, filters_output = n_filters, filters_hidden = nh_filters, kernel_size = 3);
    
    t = tf.keras.layers.MaxPooling2D(pool_size=2)(t);
    
    ########
    
    t = residual_block1( t, filters_output = n_filters, filters_hidden = nh_filters, kernel_size = 3);
    
    t = residual_block1( t, filters_output = n_filters, filters_hidden = nh_filters, kernel_size = 3);
    
    t = tf.keras.layers.MaxPooling2D(pool_size=2)(t);
    
    ########
    
    t = tf.keras.layers.Flatten()(t);
    
    outputs = tf.keras.layers.Dense(Nout, activation='tanh')(t);
    
    model = tf.keras.models.Model(inputs, outputs);
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy']);
    
    if enable_summary:
        model.summary();
    
    return model, target_size;
    
def create_model_custom_inception(file_of_weight='',tuning_feature_extractor=True):
    '''
    Retorna un modelo para la clasificación.
    Adicionalmente, si el archivo `file_of_weight` existe los pesos son cargados.
    
    :param file_of_weight: Archivo donde se encuentran los pesos.
    :type file_of_weight: str
    :return: Retorna un modelo de red neuronal
    :rtype: tensorflow.python.keras.engine.sequential.Sequential
    '''
    
    target_size=(299,299);
    layer_inception = tf.keras.applications.InceptionV3( include_top=False, weights="imagenet", 
                                                         input_shape=(target_size[0],target_size[1],3));
    
    #layer_inception.trainable=False
    

    
    modelo = tf.keras.Sequential([
        layer_inception,
        tf.keras.layers.Conv2D( 16, kernel_size=1, padding="same", activation='relu'),
        
        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(32,activation='tanh'),

    ])
    
    modelo.layers[0].trainable = tuning_feature_extractor
    
    print("\n\n")
    modelo.summary();
    
    return modelo, target_size
    
def create_model_custom1(file_of_weight=''):
    '''
    Retorna un modelo para la clasificación.
    Adicionalmente, si el archivo `file_of_weight` existe los pesos son cargados.
    
    :param file_of_weight: Archivo donde se encuentran los pesos.
    :type file_of_weight: str
    :return: Retorna un modelo de red neuronal
    :rtype: tensorflow.python.keras.engine.sequential.Sequential
    '''
    target_size=(224,224);
    
    # modelo nuevo
    modelo = tf.keras.Sequential([
        tf.keras.layers.Conv2D( 16, kernel_size=11, padding="same", activation='relu',
                                input_shape=(target_size[0],target_size[1],3)),
        tf.keras.layers.Conv2D(  4, kernel_size=9, padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D( pool_size=(2, 2)),
        
        tf.keras.layers.Conv2D( 16, kernel_size=9, padding="same", activation='relu'),
        tf.keras.layers.Conv2D(  4, kernel_size=7, padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D( pool_size=(2, 2)),
        
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Conv2D( 16, kernel_size=7, padding="same", activation='relu'),
        tf.keras.layers.Conv2D(  4, kernel_size=5, padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D( pool_size=(2, 2)),
        
        tf.keras.layers.Conv2D( 16, kernel_size=5, padding="same", activation='relu'),
        tf.keras.layers.Conv2D(  4, kernel_size=3, padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D( pool_size=(2, 2)),
        
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Conv2D( 16, kernel_size=3, padding="same", activation='relu'),
        tf.keras.layers.Conv2D(  4, kernel_size=3, padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D( pool_size=(2, 2)),
        
        tf.keras.layers.Conv2D( 16, kernel_size=3, padding="same", activation='relu'),
        tf.keras.layers.Conv2D( 8, kernel_size=3, padding="same", activation='relu'),
        
        tf.keras.layers.Flatten(),
        
        #tf.keras.layers.Dense(32,activation='tanh'),

    ])
    
    modelo.summary();
    
    #if os.path.exists(file_of_weight):
    if (len(file_of_weight)!=0):
        obj=modelo.load_weights(file_of_weight);
    
    return modelo, target_size
    
    
def create_model(file_of_weight='',model_type='mobilenet_v3',load_weights=False,nout=7,tuning_feature_extractor = True):
    '''
    Retorna un modelo para la clasificación.
    Adicionalmente, si el archivo `file_of_weight` existe los pesos son cargados.
    
    :param file_of_weight: Archivo donde se encuentran los pesos.
    :type file_of_weight: str
    :return: Retorna un modelo de red neuronal
    :rtype: tensorflow.python.keras.engine.sequential.Sequential
    '''
    url=None;
    
    if   model_type=='mobilenet_v3':
        url='https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/feature_vector/5';
        target_size=(224,224);
        multiple_layers = hub.KerasLayer(url,input_shape=(target_size[0],target_size[1],3))
        multiple_layers.trainable = tuning_feature_extractor; #False
        
    elif model_type=='resnet_v2_50':
        url='https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5';
        target_size=(224,224);
        multiple_layers = hub.KerasLayer(url,input_shape=(target_size[0],target_size[1],3))
        multiple_layers.trainable = tuning_feature_extractor; #False
        
    elif model_type=='efficientnet_b3':
        url='https://tfhub.dev/tensorflow/efficientnet/b3/feature-vector/1';
        target_size=(300,300);
        multiple_layers = hub.KerasLayer(url,input_shape=(target_size[0],target_size[1],3))
        multiple_layers.trainable = tuning_feature_extractor; #False
        
    elif model_type=='inception_v3':
        url='https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4';
        target_size=(299,299);
        multiple_layers = hub.KerasLayer(url,input_shape=(target_size[0],target_size[1],3))
        multiple_layers.trainable = tuning_feature_extractor; #False
        
    elif model_type=='inception_resnet_v2':
        url='https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/5';
        target_size=(299,299);
        multiple_layers = hub.KerasLayer(url,input_shape=(target_size[0],target_size[1],3))
        multiple_layers.trainable = tuning_feature_extractor; #False
        
    elif model_type=='custom1':
        multiple_layers, target_size = create_model_custom1(file_of_weight='');
        multiple_layers.trainable = tuning_feature_extractor; #False
        url='custom1';
        
    elif model_type=='custom_dense1':
        multiple_layers, target_size = create_model_dense1(enable_summary=True,Nout=16);
        url='custom_dense1';
        
    elif model_type=='custom_residual1':
        multiple_layers, target_size = create_model_residual1(enable_summary=True,Nout=16);
        url='custom_residual1';
        
    elif model_type=='custom_inception':
        multiple_layers, target_size = create_model_custom_inception(file_of_weight='');
        url='custom_inception';
        
    else:
        sys.exit('Error! Not found model_type: '+model_type);

    print("")
    print("Transfer learning architecture",model_type);
    print('        url:',url);
    print('target_size:',target_size);

    # modelo nuevo
    modelo = tf.keras.Sequential([
        multiple_layers,
        tf.keras.layers.Dense(32,activation='tanh'),
        tf.keras.layers.Dense(nout,activation='softmax')
    ])
    
    if load_weights==True:
        path_actual = os.path.realpath(__file__);
        directorio_actual = os.path.dirname(path_actual);
        path_of_model=os.path.join(directorio_actual,'models','model_'+model_type+'.h5');
        
        if os.path.exists(path_of_model):
            print("Loading the weights in:",path_of_model);
            try:
                modelo.load_weights(path_of_model);
                print("Loaded the weights in:",path_of_model);
                
            except Exception:
                print("Error loading the weights in:",path_of_model);
                exit();
        else:
            print("Error loading, file no found:",path_of_model);
       
    if len(file_of_weight)!=0:
        print("Loading the weights in:",file_of_weight);
        if os.path.exists(file_of_weight):
            #
            try:
                obj=modelo.load_weights(file_of_weight);
                print("Loaded the weights in:",file_of_weight);
            except Exception:
                print("Error loading the weights in:",file_of_weight);
                exit();
        else:
            print("Error loading, file no found:",file_of_weight);
    
    
    return modelo, target_size

def evaluate_model_from_file(modelo, imgfilepath,target_size=(224,224)):
    '''
    Evalua la red neuronal descrita en `modelo`, la entrada es leida desde el archivo `imgfilepath`.
    
    :param modelo: Modelo de la red neuronal.
    :type modelo: tensorflow.python.keras.engine.sequential.Sequential
    :param imgfilepath: Archivo de donde se leerá la imagen a testar.
    :type imgfilepath: str
    :return: Retorna la classificación, 
    :rtype: integer
    '''
    image = load_img(imgfilepath)
    image = img_to_array(image)
    image=cv2.resize(image,target_size);
    image = np.expand_dims(image, axis=0)
    image=image/255.0;
    res=modelo.predict(image.reshape(-1,target_size[0],target_size[1],3),verbose=0);
    #print(res)
    #print(res.shape)
    return np.argmax(res);

def evaluate_model_from_pil(modelo, image,target_size=(224,224)):
    '''
    Evalua la red neuronal descrita en `modelo`, la entrada es leida desde una imagen PIL.
    
    :param modelo: Modelo de la red neuronal.
    :type modelo: tensorflow.python.keras.engine.sequential.Sequential
    :param image: Imagen a testar.
    :type image: PIL.PngImagePlugin.PngImageFile
    :return: Retorna la classificación.
    :rtype: bool
    '''
    image=np.array(image)
    image=cv2.resize(image,target_size);
    image = np.expand_dims(image, axis=0)
    image=image/255.0;
    res=modelo.predict(image.reshape(-1,target_size[0],target_size[1],3),verbose=0);
    #print(res)
    #print(res.shape)
    return np.argmax(res);
    
def predict_from_pil(modelo, image,target_size=(224,224)):
    '''
    Evalua la red neuronal descrita en `modelo`, la entrada es leida desde una imagen PIL.
    
    :param modelo: Modelo de la red neuronal.
    :type modelo: tensorflow.python.keras.engine.sequential.Sequential
    :param image: Imagen a testar.
    :type image: PIL.PngImagePlugin.PngImageFile
    :return: Retorna la classificación.
    :rtype: bool
    '''
    image=np.array(image)
    image=cv2.resize(image,target_size);
    image = np.expand_dims(image, axis=0)
    image=image/255.0;
    res=modelo.predict(image.reshape(-1,target_size[0],target_size[1],3),verbose=0);
    #print(res)
    #print(res.shape)
    return res[0];

def predict_from_pil_list(modelo, pil_list, target_size=(224,224)):
    '''
    Evalua la red neuronal descrita en `modelo`, la entrada es leida desde una imagen PIL.
    
    :param modelo: Modelo de la red neuronal.
    :type modelo: tensorflow.python.keras.engine.sequential.Sequential
    :param pil_list: List of Imagen a testar.
    :type pil_list: List of PIL.PngImagePlugin.PngImageFile
    :return: Retorna la matriz classificación.
    :rtype: numpy.ndarray
    '''
    L=len(pil_list);
    
    image_array = np.zeros((L, target_size[0], target_size[1], 3), dtype=np.float32);

    none_id_list=[];
    for i, img in enumerate(pil_list):
        if img is None:
            none_id_list.append(i);
        else:
            img_resized=cv2.resize(np.array(img),target_size);
            image_array[i] = img_resized / 255.0;
        
    res=modelo.predict(image_array,verbose=0);
    
    # Zerando as linhas correspondentes aos IDs
    res[none_id_list, :] = 0.0;
    
    return res;

def evaluate_model_from_pil_list(modelo, pil_list,target_size=(224,224)):
    '''
    Evalua la red neuronal descrita en `modelo`, la entrada es leida desde una imagen PIL.
    
    :param modelo: Modelo de la red neuronal.
    :type modelo: tensorflow.python.keras.engine.sequential.Sequential
    :param target_size: List of Imagen a testar.
    :type target_size: List of PIL.PngImagePlugin.PngImageFile
    :return: Retorna la classificación.
    :rtype: vector
    '''
    
    return np.argmax(predict_from_pil_list(modelo, pil_list,target_size=target_size),axis=1);

def save_model_history(hist, fpath,show=True, labels=['accuracy','loss']):
    ''''This function saves the history returned by model.fit to a tab-
    delimited file, where model is a keras model'''

    acc      = hist.history[labels[0]];
    val_acc  = hist.history['val_'+labels[0]];
    loss     = hist.history[labels[1]];
    val_loss = hist.history['val_'+labels[1]];

    EPOCAS=len(acc);
    
    rango_epocas=range(EPOCAS);

    plt.figure(figsize=(16,8))
    #
    plt.subplot(1,2,1)
    plt.plot(rango_epocas,    acc,label=labels[0]+' training')
    plt.plot(rango_epocas,val_acc,label=labels[0]+' validation')
    plt.legend(loc='lower right')
    #plt.title('Analysis accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    #
    plt.subplot(1,2,2)
    plt.plot(rango_epocas,    loss,label=labels[1]+' training')
    plt.plot(rango_epocas,val_loss,label=labels[1]+' validation')
    plt.legend(loc='lower right')
    #plt.title('Analysis loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    #
    plt.savefig(fpath+'.plot.png')
    if show:
        plt.show()
    
    print('max_val_acc', np.max(val_acc))
    
    ###########
    
    # Open file
    fid = open(fpath, 'w')
    print('accuracy,val_accuracy,loss,val_loss', file = fid)

    try:
        # Iterate through
        for i in rango_epocas:
            print('{},{},{},{}'.format(acc[i],val_acc[i],loss[i],val_loss[i]),file = fid)
    except KeyError:
        print('<no history found>', file = fid)

    # Close file
    fid.close()
    
    return acc, val_acc

def save_model_stat_kfold(VALIDATION_ACCURACY,VALIDATION_LOSS, fpath):
    '''
    Salva los datos de accuracy y loss en un archivo de tipo m.
    
    :param VALIDATION_ACCURACY: Lista de accuracies
    :type VALIDATION_ACCURACY: list of floats
    :param VALIDATION_LOSS: Lista de loss
    :type VALIDATION_LOSS: list of floats
    :param fpath: Archivo donde se guardaran los datos.
    :type fpath: str
    :return: Retorna el valor medio de las acuracias.
    :rtype: float
    '''
    fid = open(fpath, 'w')
    
    #
    print('mean_val_acc={}'.format(np.mean(VALIDATION_ACCURACY)),';', file = fid)
    
    #
    print('std_val_acc={}'.format(np.std(VALIDATION_ACCURACY)),';', file = fid)
    
    #
    print('mean_val_loss={}'.format(np.mean(VALIDATION_LOSS)),';', file = fid)
    
    #
    print('std_val_loss={}'.format(np.std(VALIDATION_LOSS)),';', file = fid)
    
    #
    print('val_acc=[', end='', file = fid)
    k=1;
    for value in VALIDATION_ACCURACY:
        if k==len(VALIDATION_ACCURACY):
            print('{}'.format(value),end='', file = fid);
        else:
            print('{}'.format(value),end=';', file = fid);
        k=k+1;
    print('];', file = fid)
    
    #
    print('val_loss=[', end='', file = fid)
    k=1;
    for value in VALIDATION_LOSS:
        if k==len(VALIDATION_LOSS):
            print('{}'.format(value),end='', file = fid);
        else:
            print('{}'.format(value),end=';', file = fid);
        k=k+1;
    print('];', file = fid)
    
    fid.close()
    return np.mean(VALIDATION_ACCURACY);


def get_model_parameters(model):
    return model.count_params();

from tensorflow.python.keras.utils.layer_utils import count_params
def save_model_parameters(model, fpath):
    '''
    Salva en un archivo la estadistica de la cantidoda de parametros de un modelo
    
    :param model: Modelos a analizar
    :type model: str
    :param fpath: Archivo donde se salvaran los datos.
    :type fpath: str
    '''
    trainable_count = count_params(model.trainable_weights)
    
    fid = open(fpath, 'w')
    print('parameters_total={}'.format(model.count_params()),';', file = fid);
    print('parameters_trainable={}'.format(trainable_count),';', file = fid);
    fid.close()
    
if __name__=="__main__":
    # export TF_USE_LEGACY_KERAS=1; python3 lib_model.py
    import gdown
    import zipfile
    import os
    
    pesos = {
        "ber2024" : "https://drive.google.com/uc?id=10PZUfBSJt3FXcNaA8UfvP6hGC46E0NoR",
        "full2024": "https://drive.google.com/uc?id=18ZTsD3FF0_1H3goacGPZwOgcLOKXhw0b"
    }
    
    OUTPUT = 'output'
    
    os.makedirs(OUTPUT, exist_ok = True)
    for key,url in pesos.items():
        print(key,url)
    
        output = 'arquivo.zip'
        weights_dir = 'weights_'+key
        
        gdown.download(url, output, quiet=False)
        
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(weights_dir)
        
        os.remove(output)
        
        os.makedirs(os.path.join(OUTPUT,"complete_"+key), exist_ok = True)
        
        name_model = ['mobilenet_v3', 'resnet_v2_50','efficientnet_b3', 'inception_v3', 'inception_resnet_v2']
        for name in name_model:
            print("\n")
            
            weights_path = os.path.join(weights_dir,'model_'+name+'.h5')
            
            model, target_size = create_model(  model_type=name,
                                                load_weights=False,
                                                nout=4,
                                                file_of_weight=weights_path)

            # Saving models
            keras_path = os.path.join(OUTPUT,"complete_"+key,"complete_"+name+".keras")
            model.save(keras_path, save_format="keras")

            st_path = os.path.join(OUTPUT,"complete_"+key,"complete_"+name)
            model.save(st_path, save_format="tf")

            # Loading models
            modelo_keras = tf.keras.models.load_model(keras_path, custom_objects={"KerasLayer": hub.KerasLayer})
            modelo_keras.load_weights(weights_path)            
            
            modelo_tf = tf.keras.models.load_model(st_path)
            modelo_tf.load_weights(weights_path)

            # print values
            print(model.layers[0].get_weights()[0][0:3])
            print(modelo_keras.layers[0].get_weights()[0][0:3])
            print(modelo_tf.layers[0].get_weights()[0][0:3])

            # Sumary
            model.summary()
            modelo_keras.summary()
            modelo_tf.summary()
