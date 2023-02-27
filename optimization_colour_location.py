# get array of N neurons with the highest changes in their activation for changes in relevant information

# Johanna Meyer


## general declarations

import numpy as np
from extract_features_optimization import extract_features

#parameters

NF_used_per = 0.0002

NF = 0 # to be specified after loading image in vgg16

NF_used = 0 # to be specified later
#test 1 for colour
#test 5 for location
images_path='images/test/test1/'

image_11=extract_features(images_path + 'circle_blue.jpg')
image_12=extract_features(images_path + 'circle_green.jpg')
image_13=extract_features(images_path + 'circle_red.jpg')
image_14=extract_features(images_path + 'circle_white.jpg')
image_15=extract_features(images_path + 'circle_yellow.jpg')
image_21=extract_features(images_path + 'rectangle_blue.jpg')
image_22=extract_features(images_path + 'rectangle_green.jpg')
image_23=extract_features(images_path + 'rectangle_red.jpg')
image_24=extract_features(images_path + 'rectangle_white.jpg')
image_25=extract_features(images_path + 'rectangle_yellow.jpg')
image_31=extract_features(images_path + 'star_blue.jpg')
image_32=extract_features(images_path + 'star_green.jpg')
image_33=extract_features(images_path + 'star_red.jpg')
image_34=extract_features(images_path + 'star_white.jpg')
image_35=extract_features(images_path + 'star_yellow.jpg')
image_41=extract_features(images_path + 'triangle_blue.jpg')
image_42=extract_features(images_path + 'triangle_green.jpg')
image_43=extract_features(images_path + 'triangle_red.jpg')
image_44=extract_features(images_path + 'triangle_white.jpg')
image_45=extract_features(images_path + 'triangle_yellow.jpg')


sub_abs=np.absolute(np.subtract(image_11,image_12))
sub_abs += np.absolute(np.subtract(image_11,image_13))
sub_abs += np.absolute(np.subtract(image_11,image_14))
sub_abs += np.absolute(np.subtract(image_11,image_15))
sub_abs += np.absolute(np.subtract(image_12,image_13))
sub_abs += np.absolute(np.subtract(image_12,image_14))
sub_abs += np.absolute(np.subtract(image_12,image_15))
sub_abs += np.absolute(np.subtract(image_13,image_14))
sub_abs += np.absolute(np.subtract(image_13,image_15))
sub_abs += np.absolute(np.subtract(image_14,image_15))

sub_abs += np.absolute(np.subtract(image_21,image_22))
sub_abs += np.absolute(np.subtract(image_21,image_23))
sub_abs += np.absolute(np.subtract(image_21,image_24))
sub_abs += np.absolute(np.subtract(image_21,image_25))
sub_abs += np.absolute(np.subtract(image_22,image_23))
sub_abs += np.absolute(np.subtract(image_22,image_24))
sub_abs += np.absolute(np.subtract(image_22,image_25))
sub_abs += np.absolute(np.subtract(image_23,image_24))
sub_abs += np.absolute(np.subtract(image_23,image_25))
sub_abs += np.absolute(np.subtract(image_24,image_25))

sub_abs += np.absolute(np.subtract(image_31,image_32))
sub_abs += np.absolute(np.subtract(image_31,image_33))
sub_abs += np.absolute(np.subtract(image_31,image_34))
sub_abs += np.absolute(np.subtract(image_31,image_35))
sub_abs += np.absolute(np.subtract(image_32,image_33))
sub_abs += np.absolute(np.subtract(image_32,image_34))
sub_abs += np.absolute(np.subtract(image_32,image_35))
sub_abs += np.absolute(np.subtract(image_33,image_34))
sub_abs += np.absolute(np.subtract(image_33,image_35))
sub_abs += np.absolute(np.subtract(image_34,image_35))

sub_abs += np.absolute(np.subtract(image_41,image_42))
sub_abs += np.absolute(np.subtract(image_41,image_43))
sub_abs += np.absolute(np.subtract(image_41,image_44))
sub_abs += np.absolute(np.subtract(image_41,image_45))
sub_abs += np.absolute(np.subtract(image_42,image_43))
sub_abs += np.absolute(np.subtract(image_42,image_44))
sub_abs += np.absolute(np.subtract(image_42,image_45))
sub_abs += np.absolute(np.subtract(image_43,image_44))
sub_abs += np.absolute(np.subtract(image_43,image_45))
sub_abs += np.absolute(np.subtract(image_44,image_45))

NF = len(sub_abs)
NF_used = int(NF_used_per * NF)
x = np.argsort(sub_abs)[::-1][:NF_used]
np.save('indices_features.npy', x)