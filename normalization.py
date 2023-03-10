from numpy import *
#from extract_features_layer_vgg16 import extract_features
from keras.applications.vgg16 import VGG16
from keras_preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from numpy import *

def extract_features(image_path):
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
    img=image.load_img(image_path,target_size=(224,224))
    x=image.img_to_array(img)
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
    return features_layer_all

image_1=extract_features('images/test/normalization/flower_.jpg')
image_2=extract_features('images/test/normalization/car.jpg')
image_3=extract_features('images/test/normalization/flower.jpg')
image_4=extract_features('images/test/normalization/house.jpg')
image_5=extract_features('images/test/normalization/dog.jpg')
image_6=extract_features('images/test/normalization/dog_.jpg')
image_7=extract_features('images/test/normalization/usain.jpg')
image_8=extract_features('images/test/normalization/circle_yellow_complexe_cue.jpg')
image_9=extract_features('images/test/normalization/mooneyfaces.jpg')
image_10=extract_features('images/test/normalization/yellow_shape.jpg')
image_11=extract_features('images/test/normalization/cologne.jpg')
image_12=extract_features('images/test/normalization/curry.jpg')
image_13=extract_features('images/test/normalization/cr7.jpg')
image_14=extract_features('images/test/normalization/bike.jpeg')
image_15=extract_features('images/test/normalization/benchpress.jpg')
image_16=extract_features('images/test/normalization/rocket.jpg')
image_17=extract_features('images/test/normalization/clock.jpg')
image_18=extract_features('images/test/normalization/a380.jpg')
image_19=extract_features('images/test/normalization/mainhattan.jpg')
image_20=extract_features('images/test/normalization/tennisball.jpg')
image_21=extract_features('images/test/normalization/audi.jpg')
image_22=extract_features('images/test/normalization/landscape.jpg')
image_23=extract_features('images/test/normalization/guitar.jpg')
image_24=extract_features('images/test/normalization/factory.jpg')
image_25=extract_features('images/test/normalization/tree.jpg')
image_26=extract_features('images/test/normalization/tomato.jpg')
image_27=extract_features('images/test/normalization/hardy.jpg')
image_28=extract_features('images/test/normalization/pizza.jpg')
image_29=extract_features('images/test/normalization/camera.jpg')
image_30=extract_features('images/test/normalization/stones.jpg')
image_31=extract_features('images/test/normalization/chicken.jpg')
image_32=extract_features('images/test/normalization/suit.jpg')
image_33=extract_features('images/test/normalization/runner.jpg')
image_34=extract_features('images/test/normalization/cruise.jpg')
image_35=extract_features('images/test/normalization/lava.jpg')
image_36=extract_features('images/test/normalization/apples.jpg')
image_37=extract_features('images/test/normalization/beats.jpg')
image_38=extract_features('images/test/normalization/bestmovie.jpg')
image_39=extract_features('images/test/normalization/dinner.jpg')
image_40=extract_features('images/test/normalization/fish.jpg')
image_41=extract_features('images/test/normalization/bees.jpg')
image_42=extract_features('images/test/normalization/ninja.jpg')
image_43=extract_features('images/test/normalization/mariokart.jpg')
image_44=extract_features('images/test/normalization/kawai.jpg')
image_45=extract_features('images/test/normalization/rollo.jpg')
image_46=extract_features('images/test/normalization/mando.jpg')
image_47=extract_features('images/test/normalization/bus.jpg')
image_48=extract_features('images/test/normalization/teuer.jpg')
image_49=extract_features('images/test/normalization/idea.jpg')
image_50=extract_features('images/test/normalization/rudolph.jpg')

a = concatenate((image_1.reshape(-1,1),image_2.reshape(-1,1)),axis=1)
a = concatenate((a,image_3.reshape(-1,1)),axis=1)
a = concatenate((a,image_4.reshape(-1,1)),axis=1)
a = concatenate((a,image_5.reshape(-1,1)),axis=1)
a = concatenate((a,image_6.reshape(-1,1)),axis=1)
a = concatenate((a,image_7.reshape(-1,1)),axis=1)
a = concatenate((a,image_8.reshape(-1,1)),axis=1)
a = concatenate((a,image_9.reshape(-1,1)),axis=1)
a = concatenate((a,image_10.reshape(-1,1)),axis=1)
a = concatenate((a,image_11.reshape(-1,1)),axis=1)
a = concatenate((a,image_12.reshape(-1,1)),axis=1)
a = concatenate((a,image_13.reshape(-1,1)),axis=1)
a = concatenate((a,image_14.reshape(-1,1)),axis=1)
a = concatenate((a,image_15.reshape(-1,1)),axis=1)
a = concatenate((a,image_16.reshape(-1,1)),axis=1)
a = concatenate((a,image_17.reshape(-1,1)),axis=1)
a = concatenate((a,image_18.reshape(-1,1)),axis=1)
a = concatenate((a,image_19.reshape(-1,1)),axis=1)
a = concatenate((a,image_20.reshape(-1,1)),axis=1)
a = concatenate((a,image_21.reshape(-1,1)),axis=1)
a = concatenate((a,image_22.reshape(-1,1)),axis=1)
a = concatenate((a,image_23.reshape(-1,1)),axis=1)
a = concatenate((a,image_24.reshape(-1,1)),axis=1)
a = concatenate((a,image_25.reshape(-1,1)),axis=1)
a = concatenate((a,image_26.reshape(-1,1)),axis=1)
a = concatenate((a,image_27.reshape(-1,1)),axis=1)
a = concatenate((a,image_28.reshape(-1,1)),axis=1)
a = concatenate((a,image_29.reshape(-1,1)),axis=1)
a = concatenate((a,image_30.reshape(-1,1)),axis=1)
a = concatenate((a,image_31.reshape(-1,1)),axis=1)
a = concatenate((a,image_32.reshape(-1,1)),axis=1)
a = concatenate((a,image_33.reshape(-1,1)),axis=1)
a = concatenate((a,image_34.reshape(-1,1)),axis=1)
a = concatenate((a,image_35.reshape(-1,1)),axis=1)
a = concatenate((a,image_36.reshape(-1,1)),axis=1)
a = concatenate((a,image_37.reshape(-1,1)),axis=1)
a = concatenate((a,image_38.reshape(-1,1)),axis=1)
a = concatenate((a,image_39.reshape(-1,1)),axis=1)
a = concatenate((a,image_40.reshape(-1,1)),axis=1)
a = concatenate((a,image_41.reshape(-1,1)),axis=1)
a = concatenate((a,image_42.reshape(-1,1)),axis=1)
a = concatenate((a,image_43.reshape(-1,1)),axis=1)
a = concatenate((a,image_44.reshape(-1,1)),axis=1)
a = concatenate((a,image_45.reshape(-1,1)),axis=1)
a = concatenate((a,image_46.reshape(-1,1)),axis=1)
a = concatenate((a,image_47.reshape(-1,1)),axis=1)
a = concatenate((a,image_48.reshape(-1,1)),axis=1)
a = concatenate((a,image_49.reshape(-1,1)),axis=1)
a = concatenate((a,image_50.reshape(-1,1)),axis=1)


neurons_mean=mean(a, axis=1)
save('normalization.npy', neurons_mean)