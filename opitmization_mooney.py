# get array of N neurons with the highest changes in their activation for changes in relevant information

# Johanna Meyer


## general declarations

import numpy as np
from extract_features_optimization import extract_features

#parameters

NF_used_per = 0.0002

NF = 0 # to be specified after loading image in vgg16

NF_used = 0 # to be specified later
images_path='images/test/test7/'
image_10=extract_features(images_path + 'mooney1.jpg')
image_11=extract_features(images_path + 'mooney2.jpg')
image_12=extract_features(images_path + 'mooney3.jpg')
image_13=extract_features(images_path + 'mooney4.jpg')
image_14=extract_features(images_path + 'mooney5.jpg')
image_15=extract_features(images_path + 'mooney6.jpg')
image_16=extract_features(images_path + 'mooney7.jpg')
image_17=extract_features(images_path + 'mooney8.jpg')
image_18=extract_features(images_path + 'mooney9.jpg')
image_19=extract_features(images_path + 'mooney10.jpg')
image_20=extract_features(images_path + 'no_mooney1.jpg')
image_21=extract_features(images_path + 'no_mooney2.jpg')
image_22=extract_features(images_path + 'no_mooney3.jpg')
image_23=extract_features(images_path + 'no_mooney4.jpg')
image_24=extract_features(images_path + 'no_mooney5.jpg')
image_25=extract_features(images_path + 'no_mooney6.jpg')
image_26=extract_features(images_path + 'no_mooney7.jpg')
image_27=extract_features(images_path + 'no_mooney8.jpg')
image_28=extract_features(images_path + 'no_mooney9.jpg')
image_29=extract_features(images_path + 'no_mooney10.jpg')

sub_abs=np.absolute(np.subtract(image_10,image_20))
sub_abs += np.absolute(np.subtract(image_11,image_21))
sub_abs += np.absolute(np.subtract(image_12,image_22))
sub_abs += np.absolute(np.subtract(image_13,image_23))
sub_abs += np.absolute(np.subtract(image_14,image_24))
sub_abs += np.absolute(np.subtract(image_15,image_25))
sub_abs += np.absolute(np.subtract(image_16,image_26))
sub_abs += np.absolute(np.subtract(image_17,image_27))
sub_abs += np.absolute(np.subtract(image_18,image_28))
sub_abs += np.absolute(np.subtract(image_19,image_29))

sub_abs += np.absolute(np.subtract(image_10,image_21))
sub_abs += np.absolute(np.subtract(image_11,image_22))
sub_abs += np.absolute(np.subtract(image_12,image_23))
sub_abs += np.absolute(np.subtract(image_13,image_24))
sub_abs += np.absolute(np.subtract(image_14,image_25))
sub_abs += np.absolute(np.subtract(image_15,image_26))
sub_abs += np.absolute(np.subtract(image_16,image_27))
sub_abs += np.absolute(np.subtract(image_17,image_28))
sub_abs += np.absolute(np.subtract(image_18,image_29))
sub_abs += np.absolute(np.subtract(image_19,image_20))

sub_abs += np.absolute(np.subtract(image_10,image_22))
sub_abs += np.absolute(np.subtract(image_11,image_23))
sub_abs += np.absolute(np.subtract(image_12,image_24))
sub_abs += np.absolute(np.subtract(image_13,image_25))
sub_abs += np.absolute(np.subtract(image_14,image_26))
sub_abs += np.absolute(np.subtract(image_15,image_27))
sub_abs += np.absolute(np.subtract(image_16,image_28))
sub_abs += np.absolute(np.subtract(image_17,image_29))
sub_abs += np.absolute(np.subtract(image_18,image_20))
sub_abs += np.absolute(np.subtract(image_19,image_21))

sub_abs += np.absolute(np.subtract(image_10,image_23))
sub_abs += np.absolute(np.subtract(image_11,image_24))
sub_abs += np.absolute(np.subtract(image_12,image_25))
sub_abs += np.absolute(np.subtract(image_13,image_26))
sub_abs += np.absolute(np.subtract(image_14,image_27))
sub_abs += np.absolute(np.subtract(image_15,image_28))
sub_abs += np.absolute(np.subtract(image_16,image_29))
sub_abs += np.absolute(np.subtract(image_17,image_20))
sub_abs += np.absolute(np.subtract(image_18,image_21))
sub_abs += np.absolute(np.subtract(image_19,image_22))

sub_abs += np.absolute(np.subtract(image_10,image_24))
sub_abs += np.absolute(np.subtract(image_11,image_25))
sub_abs += np.absolute(np.subtract(image_12,image_26))
sub_abs += np.absolute(np.subtract(image_13,image_27))
sub_abs += np.absolute(np.subtract(image_14,image_28))
sub_abs += np.absolute(np.subtract(image_15,image_29))
sub_abs += np.absolute(np.subtract(image_16,image_20))
sub_abs += np.absolute(np.subtract(image_17,image_21))
sub_abs += np.absolute(np.subtract(image_18,image_22))
sub_abs += np.absolute(np.subtract(image_19,image_23))

sub_abs += np.absolute(np.subtract(image_10,image_25))
sub_abs += np.absolute(np.subtract(image_11,image_26))
sub_abs += np.absolute(np.subtract(image_12,image_27))
sub_abs += np.absolute(np.subtract(image_13,image_28))
sub_abs += np.absolute(np.subtract(image_14,image_29))
sub_abs += np.absolute(np.subtract(image_15,image_20))
sub_abs += np.absolute(np.subtract(image_16,image_21))
sub_abs += np.absolute(np.subtract(image_17,image_22))
sub_abs += np.absolute(np.subtract(image_18,image_23))
sub_abs += np.absolute(np.subtract(image_19,image_24))

sub_abs += np.absolute(np.subtract(image_10,image_25))
sub_abs += np.absolute(np.subtract(image_11,image_26))
sub_abs += np.absolute(np.subtract(image_12,image_27))
sub_abs += np.absolute(np.subtract(image_13,image_28))
sub_abs += np.absolute(np.subtract(image_14,image_29))
sub_abs += np.absolute(np.subtract(image_15,image_20))
sub_abs += np.absolute(np.subtract(image_16,image_21))
sub_abs += np.absolute(np.subtract(image_17,image_22))
sub_abs += np.absolute(np.subtract(image_18,image_23))
sub_abs += np.absolute(np.subtract(image_19,image_24))

sub_abs += np.absolute(np.subtract(image_10,image_26))
sub_abs += np.absolute(np.subtract(image_11,image_27))
sub_abs += np.absolute(np.subtract(image_12,image_28))
sub_abs += np.absolute(np.subtract(image_13,image_29))
sub_abs += np.absolute(np.subtract(image_14,image_20))
sub_abs += np.absolute(np.subtract(image_15,image_21))
sub_abs += np.absolute(np.subtract(image_16,image_22))
sub_abs += np.absolute(np.subtract(image_17,image_23))
sub_abs += np.absolute(np.subtract(image_18,image_24))
sub_abs += np.absolute(np.subtract(image_19,image_25))

sub_abs += np.absolute(np.subtract(image_10,image_27))
sub_abs += np.absolute(np.subtract(image_11,image_28))
sub_abs += np.absolute(np.subtract(image_12,image_29))
sub_abs += np.absolute(np.subtract(image_13,image_20))
sub_abs += np.absolute(np.subtract(image_14,image_21))
sub_abs += np.absolute(np.subtract(image_15,image_22))
sub_abs += np.absolute(np.subtract(image_16,image_23))
sub_abs += np.absolute(np.subtract(image_17,image_24))
sub_abs += np.absolute(np.subtract(image_18,image_25))
sub_abs += np.absolute(np.subtract(image_19,image_26))

sub_abs += np.absolute(np.subtract(image_10,image_28))
sub_abs += np.absolute(np.subtract(image_11,image_29))
sub_abs += np.absolute(np.subtract(image_12,image_20))
sub_abs += np.absolute(np.subtract(image_13,image_21))
sub_abs += np.absolute(np.subtract(image_14,image_22))
sub_abs += np.absolute(np.subtract(image_15,image_23))
sub_abs += np.absolute(np.subtract(image_16,image_24))
sub_abs += np.absolute(np.subtract(image_17,image_25))
sub_abs += np.absolute(np.subtract(image_18,image_26))
sub_abs += np.absolute(np.subtract(image_19,image_27))

sub_abs += np.absolute(np.subtract(image_10,image_29))
sub_abs += np.absolute(np.subtract(image_11,image_20))
sub_abs += np.absolute(np.subtract(image_12,image_21))
sub_abs += np.absolute(np.subtract(image_13,image_22))
sub_abs += np.absolute(np.subtract(image_14,image_23))
sub_abs += np.absolute(np.subtract(image_15,image_24))
sub_abs += np.absolute(np.subtract(image_16,image_25))
sub_abs += np.absolute(np.subtract(image_17,image_26))
sub_abs += np.absolute(np.subtract(image_18,image_27))
sub_abs += np.absolute(np.subtract(image_19,image_28))


NF = len(sub_abs)
NF_used = int(NF_used_per * NF)
x = np.argsort(sub_abs)[::-1][:NF_used]
np.save('indices_features.npy', x)