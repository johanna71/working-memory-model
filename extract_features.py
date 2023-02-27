#extract_features_layer_vgg16.py
import tensorflow
from tensorflow import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras import utils
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from numpy import *
from sklearn import preprocessing

def extract_features(image_path):
    normalization_array= load('normalization.npy')
    normalization_array[normalization_array==0]=1000000000
    print(image_path)
    base_model=VGG16(weights='imagenet')
    model_layer_1=Model(inputs=base_model.input, outputs = base_model.layers[1].output)
    model_layer_2=Model(inputs=base_model.input, outputs = base_model.layers[2].output)
    model_layer_4=Model(inputs=base_model.input, outputs = base_model.layers[4].output)
    model_layer_5=Model(inputs=base_model.input, outputs = base_model.layers[5].output)
    model_layer_7=Model(inputs=base_model.input, outputs = base_model.layers[7].output)
    model_layer_8=Model(inputs=base_model.input, outputs = base_model.layers[8].output)
    model_layer_9=Model(inputs=base_model.input, outputs = base_model.layers[9].output)
    model_layer_11=Model(inputs=base_model.input, outputs = base_model.layers[11].output)
    model_layer_12=Model(inputs=base_model.input, outputs = base_model.layers[12].output)
    model_layer_13=Model(inputs=base_model.input, outputs = base_model.layers[13].output)
    model_layer_15=Model(inputs=base_model.input, outputs = base_model.layers[15].output)
    model_layer_16=Model(inputs=base_model.input, outputs = base_model.layers[16].output)
    model_layer_17=Model(inputs=base_model.input, outputs = base_model.layers[17].output)
    model_layer_20=Model(inputs=base_model.input, outputs = base_model.layers[20].output)
    model_layer_21=Model(inputs=base_model.input, outputs = base_model.layers[21].output)
    model_layer_22=Model(inputs=base_model.input, outputs = base_model.layers[22].output)
    img=utils.load_img(image_path,target_size=(224,224))
    x=utils.img_to_array(img)
    x=expand_dims(x,axis=0)
    x=preprocess_input(x)
    features_layer_1=model_layer_1.predict(x)
    features_layer_2=model_layer_2.predict(x)
    features_layer_all_1=append(features_layer_2,features_layer_1)
    features_layer_4=model_layer_4.predict(x)
    features_layer_all_2=append(features_layer_4,features_layer_all_1)
    features_layer_5=model_layer_5.predict(x)
    features_layer_all_3=append(features_layer_5,features_layer_all_2)
    features_layer_7=model_layer_7.predict(x)
    features_layer_all_4=append(features_layer_7,features_layer_all_3)
    features_layer_8=model_layer_8.predict(x)
    features_layer_all_5=append(features_layer_8,features_layer_all_4)
    features_layer_9=model_layer_9.predict(x)
    features_layer_all_6=append(features_layer_9,features_layer_all_5)
    features_layer_11=model_layer_11.predict(x)
    features_layer_all_7=append(features_layer_11,features_layer_all_6)
    features_layer_12=model_layer_12.predict(x)
    features_layer_all_8=append(features_layer_12,features_layer_all_7)
    features_layer_13=model_layer_13.predict(x)
    features_layer_all_9=append(features_layer_13,features_layer_all_8)
    features_layer_15=model_layer_15.predict(x)
    features_layer_all_10=append(features_layer_15,features_layer_all_9)
    features_layer_16=model_layer_16.predict(x)
    features_layer_all_11=append(features_layer_16,features_layer_all_10)
    features_layer_17=model_layer_17.predict(x)
    features_layer_all_12=append(features_layer_17,features_layer_all_11)
    features_layer_20=model_layer_20.predict(x)
    features_layer_all_13=append(features_layer_20,features_layer_all_12)
    features_layer_21=model_layer_21.predict(x)
    features_layer_all_14=append(features_layer_21,features_layer_all_13)
    features_layer_22=model_layer_22.predict(x) 
    features_layer_all=append(features_layer_22,features_layer_all_14)
    features_layer_all /= normalization_array 
    indices_array = load('indices_features.npy')
    indices = indices_array.tolist()
    indices =sort(indices)
    #indices = int(indices)
    features_layer_all=features_layer_all[indices]
    features_layer_all_normalized= preprocessing.normalize([features_layer_all])
    return features_layer_all_normalized