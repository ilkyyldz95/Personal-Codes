from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, \
    Activation, Merge, Lambda
from keras.models import Model
from googlenet_custom_layers import PoolHelper, LRN
from keras.regularizers import l2
from keras.utils import plot_model

def BTPred(scalars):
    """
    This is the output when we predict comparison labels. s1, s2 = scalars (beta.*x)
    """
    s1 = scalars[0]
    s2 = scalars[1]
    return s1 - s2

# Aim is to not require an input layer
def create_base_network(input_a, input_b, no_classes=1, no_features=None, num_score_layer=1):
    # Iniitalize all the layers
    conv1_7x7_s2 = Conv2D(64, (7, 7), name="conv1/7x7_s2", activation="relu", padding="same", strides=(2, 2),
                          kernel_regularizer=l2(0.0002))
    conv1_zero_pad = ZeroPadding2D(padding=(1, 1))
    pool1_helper = PoolHelper()
    pool1_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool1/3x3_s2')
    pool1_norm1 = LRN(name='pool1/norm1')
    conv2_3x3_reduce = Conv2D(64, (1, 1), padding='same', activation='relu', name='conv2/3x3_reduce',
                              kernel_regularizer=l2(0.0002))
    conv2_3x3 = Conv2D(192, (3, 3), padding='same', activation='relu', name='conv2/3x3', kernel_regularizer=l2(0.0002))
    conv2_norm2 = LRN(name='conv2/norm2')
    conv2_zero_pad = ZeroPadding2D(padding=(1, 1))
    pool2_helper = PoolHelper()
    pool2_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool2/3x3_s2')
    inception_3a_1x1 = Conv2D(64, (1, 1), padding='same', activation='relu', name='inception_3a/1x1',
                              kernel_regularizer=l2(0.0002))
    inception_3a_3x3_reduce = Conv2D(96, (1, 1), padding='same', activation='relu',
                                     name='inception_3a/3x3_reduce', kernel_regularizer=l2(0.0002))
    inception_3a_3x3 = Conv2D(128, (3, 3), padding='same', activation='relu', name='inception_3a/3x3',
                              kernel_regularizer=l2(0.0002))
    inception_3a_5x5_reduce = Conv2D(16, (1, 1), padding='same', activation='relu',
                                     name='inception_3a/5x5_reduce', kernel_regularizer=l2(0.0002))
    inception_3a_5x5 = Conv2D(32, (5, 5), padding='same', activation='relu', name='inception_3a/5x5',
                              kernel_regularizer=l2(0.0002))
    inception_3a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_3a/pool')
    inception_3a_pool_proj = Conv2D(32, (1, 1), padding='same', activation='relu',
                                    name='inception_3a/pool_proj', kernel_regularizer=l2(0.0002))

    inception_3b_1x1 = Conv2D(128, (1, 1), padding='same', activation='relu', name='inception_3b/1x1',
                              kernel_regularizer=l2(0.0002))
    inception_3b_3x3_reduce = Conv2D(128, (1, 1), padding='same', activation='relu',
                                     name='inception_3b/3x3_reduce', kernel_regularizer=l2(0.0002))
    inception_3b_3x3 = Conv2D(192, (3, 3), padding='same', activation='relu', name='inception_3b/3x3',
                              kernel_regularizer=l2(0.0002))
    inception_3b_5x5_reduce = Conv2D(32, (1, 1), padding='same', activation='relu',
                                     name='inception_3b/5x5_reduce', kernel_regularizer=l2(0.0002))
    inception_3b_5x5 = Conv2D(96, (5, 5), padding='same', activation='relu', name='inception_3b/5x5',
                              kernel_regularizer=l2(0.0002))
    inception_3b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_3b/pool')
    inception_3b_pool_proj = Conv2D(64, (1, 1), padding='same', activation='relu',
                                    name='inception_3b/pool_proj', kernel_regularizer=l2(0.0002))

    inception_3b_output_zero_pad = ZeroPadding2D(padding=(1, 1))
    pool3_helper = PoolHelper()
    pool3_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool3/3x3_s2')
    inception_4a_1x1 = Conv2D(192, (1, 1), padding='same', activation='relu', name='inception_4a/1x1',
                              kernel_regularizer=l2(0.0002))
    inception_4a_3x3_reduce = Conv2D(96, (1, 1), padding='same', activation='relu',
                                     name='inception_4a/3x3_reduce', kernel_regularizer=l2(0.0002))
    inception_4a_3x3 = Conv2D(208, (3, 3), padding='same', activation='relu', name='inception_4a/3x3',
                              kernel_regularizer=l2(0.0002))
    inception_4a_5x5_reduce = Conv2D(16, (1, 1), padding='same', activation='relu',
                                     name='inception_4a/5x5_reduce', kernel_regularizer=l2(0.0002))
    inception_4a_5x5 = Conv2D(48, (5, 5), padding='same', activation='relu', name='inception_4a/5x5',
                              kernel_regularizer=l2(0.0002))
    inception_4a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4a/pool')
    inception_4a_pool_proj = Conv2D(64, (1, 1), padding='same', activation='relu',
                                    name='inception_4a/pool_proj', kernel_regularizer=l2(0.0002))

    inception_4b_1x1 = Conv2D(160, (1, 1), padding='same', activation='relu', name='inception_4b/1x1',
                              kernel_regularizer=l2(0.0002))
    inception_4b_3x3_reduce = Conv2D(112, (1, 1), padding='same', activation='relu',
                                     name='inception_4b/3x3_reduce', kernel_regularizer=l2(0.0002))
    inception_4b_3x3 = Conv2D(224, (3, 3), padding='same', activation='relu', name='inception_4b/3x3',
                              kernel_regularizer=l2(0.0002))
    inception_4b_5x5_reduce = Conv2D(24, (1, 1), padding='same', activation='relu',
                                     name='inception_4b/5x5_reduce', kernel_regularizer=l2(0.0002))
    inception_4b_5x5 = Conv2D(64, (5, 5), padding='same', activation='relu', name='inception_4b/5x5',
                              kernel_regularizer=l2(0.0002))
    inception_4b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4b/pool')
    inception_4b_pool_proj = Conv2D(64, (1, 1), padding='same', activation='relu',
                                    name='inception_4b/pool_proj', kernel_regularizer=l2(0.0002))

    inception_4c_1x1 = Conv2D(128, (1, 1), padding='same', activation='relu', name='inception_4c/1x1',
                              kernel_regularizer=l2(0.0002))
    inception_4c_3x3_reduce = Conv2D(128, (1, 1), padding='same', activation='relu',
                                     name='inception_4c/3x3_reduce', kernel_regularizer=l2(0.0002))
    inception_4c_3x3 = Conv2D(256, (3, 3), padding='same', activation='relu', name='inception_4c/3x3',
                              kernel_regularizer=l2(0.0002))
    inception_4c_5x5_reduce = Conv2D(24, (1, 1), padding='same', activation='relu',
                                     name='inception_4c/5x5_reduce', kernel_regularizer=l2(0.0002))
    inception_4c_5x5 = Conv2D(64, (5, 5), padding='same', activation='relu', name='inception_4c/5x5',
                              kernel_regularizer=l2(0.0002))
    inception_4c_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4c/pool')
    inception_4c_pool_proj = Conv2D(64, (1, 1), padding='same', activation='relu',
                                    name='inception_4c/pool_proj', kernel_regularizer=l2(0.0002))

    inception_4d_1x1 = Conv2D(112, (1, 1), padding='same', activation='relu', name='inception_4d/1x1',
                              kernel_regularizer=l2(0.0002))
    inception_4d_3x3_reduce = Conv2D(144, (1, 1), padding='same', activation='relu',
                                     name='inception_4d/3x3_reduce', kernel_regularizer=l2(0.0002))
    inception_4d_3x3 = Conv2D(288, (3, 3), padding='same', activation='relu', name='inception_4d/3x3',
                              kernel_regularizer=l2(0.0002))
    inception_4d_5x5_reduce = Conv2D(32, (1, 1), padding='same', activation='relu',
                                     name='inception_4d/5x5_reduce', kernel_regularizer=l2(0.0002))
    inception_4d_5x5 = Conv2D(64, (5, 5), padding='same', activation='relu', name='inception_4d/5x5',
                              kernel_regularizer=l2(0.0002))
    inception_4d_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4d/pool')
    inception_4d_pool_proj = Conv2D(64, (1, 1), padding='same', activation='relu',
                                    name='inception_4d/pool_proj', kernel_regularizer=l2(0.0002))

    inception_4e_1x1 = Conv2D(256, (1, 1), padding='same', activation='relu', name='inception_4e/1x1',
                              kernel_regularizer=l2(0.0002))
    inception_4e_3x3_reduce = Conv2D(160, (1, 1), padding='same', activation='relu',
                                     name='inception_4e/3x3_reduce', kernel_regularizer=l2(0.0002))
    inception_4e_3x3 = Conv2D(320, (3, 3), padding='same', activation='relu', name='inception_4e/3x3',
                              kernel_regularizer=l2(0.0002))
    inception_4e_5x5_reduce = Conv2D(32, (1, 1), padding='same', activation='relu',
                                     name='inception_4e/5x5_reduce', kernel_regularizer=l2(0.0002))
    inception_4e_5x5 = Conv2D(128, (5, 5), padding='same', activation='relu', name='inception_4e/5x5',
                              kernel_regularizer=l2(0.0002))
    inception_4e_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4e/pool')
    inception_4e_pool_proj = Conv2D(128, (1, 1), padding='same', activation='relu',
                                    name='inception_4e/pool_proj', kernel_regularizer=l2(0.0002))

    inception_4e_output_zero_pad = ZeroPadding2D(padding=(1, 1))
    pool4_helper = PoolHelper()
    pool4_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool4/3x3_s2')
    inception_5a_1x1 = Conv2D(256, (1, 1), padding='same', activation='relu', name='inception_5a/1x1',
                              kernel_regularizer=l2(0.0002))
    inception_5a_3x3_reduce = Conv2D(160, (1, 1), padding='same', activation='relu',
                                     name='inception_5a/3x3_reduce', kernel_regularizer=l2(0.0002))
    inception_5a_3x3 = Conv2D(320, (3, 3), padding='same', activation='relu', name='inception_5a/3x3',
                              kernel_regularizer=l2(0.0002))
    inception_5a_5x5_reduce = Conv2D(32, (1, 1), padding='same', activation='relu',
                                     name='inception_5a/5x5_reduce', kernel_regularizer=l2(0.0002))
    inception_5a_5x5 = Conv2D(128, (5, 5), padding='same', activation='relu', name='inception_5a/5x5',
                              kernel_regularizer=l2(0.0002))
    inception_5a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_5a/pool')
    inception_5a_pool_proj = Conv2D(128, (1, 1), padding='same', activation='relu',
                                    name='inception_5a/pool_proj', kernel_regularizer=l2(0.0002))

    inception_5b_1x1 = Conv2D(384, (1, 1), padding='same', activation='relu', name='inception_5b/1x1',
                              kernel_regularizer=l2(0.0002))
    inception_5b_3x3_reduce = Conv2D(192, (1, 1), padding='same', activation='relu',
                                     name='inception_5b/3x3_reduce', kernel_regularizer=l2(0.0002))
    inception_5b_3x3 = Conv2D(384, (3, 3), padding='same', activation='relu', name='inception_5b/3x3',
                              kernel_regularizer=l2(0.0002))
    inception_5b_5x5_reduce = Conv2D(48, (1, 1), padding='same', activation='relu',
                                     name='inception_5b/5x5_reduce', kernel_regularizer=l2(0.0002))
    inception_5b_5x5 = Conv2D(128, (5, 5), padding='same', activation='relu', name='inception_5b/5x5',
                              kernel_regularizer=l2(0.0002))
    inception_5b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_5b/pool')
    inception_5b_pool_proj = Conv2D(128, (1, 1), padding='same', activation='relu',
                                    name='inception_5b/pool_proj', kernel_regularizer=l2(0.0002))

    pool5_7x7_s1 = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), name='pool5/7x7_s2')
    loss3_flat = Flatten()
    pool5_drop_7x7_s1 = Dropout(0.4, name='dropout')
    ################################################
    # Only for absolute labels
    if no_classes > 2:
        print("softmax activation")
        loss3_classifier = Dense(no_classes, name='loss3/classifier_modified', kernel_regularizer=l2(0.0002))
        loss3_classifier_act = Activation('softmax', name='prob_modified')
    else:
        print("sigmoid activation")
        loss3_classifier = Dense(1, name='loss3/classifier_modified', kernel_regularizer=l2(0.0002))
        loss3_classifier_act = Activation('sigmoid', name='prob_modified')
    ################################################
    if num_score_layer == 2:
        score_1_layer = Dense(no_features, name='score_1')
    elif num_score_layer == 3:
        score_1_layer = Dense(no_features, name='score_1')
        score_2_layer = Dense(int(no_features/2), name='score_2')
    elif num_score_layer == 4:
        score_1_layer = Dense(no_features, name='score_1')
        score_2_layer = Dense(int(no_features/2), name='score_2')
        score_3_layer = Dense(int(no_features/(2**2)), name='score_3')
    elif num_score_layer == 5:
        score_1_layer = Dense(no_features, name='score_1')
        score_2_layer = Dense(int(no_features/(2)), name='score_2')
        score_3_layer = Dense(int(no_features/(2**2)), name='score_3')
        score_4_layer = Dense(int(no_features/(2**3)), name='score_4')
    elif num_score_layer == 6:
        score_1_layer = Dense(no_features, name='score_1')
        score_2_layer = Dense(int(no_features/(2)), name='score_2')
        score_3_layer = Dense(int(no_features/(2**2)), name='score_3')
        score_4_layer = Dense(int(no_features/(2**3)), name='score_4')
        score_5_layer = Dense(int(no_features/(2**4)), name='score_5')
    score_last_layer = Dense(1, name='score_last_layer')
    #####################################
    # Siamese
    # input_b = Input(shape=(3, 224, 224))
    # Input 1 for google net.
    conv1_7x7_s2_b = conv1_7x7_s2(input_b)
    conv1_zero_pad_b = conv1_zero_pad(conv1_7x7_s2_b)
    pool1_helper_b = pool1_helper(conv1_zero_pad_b)
    pool1_3x3_s2_b = pool1_3x3_s2(pool1_helper_b)
    pool1_norm1_b = pool1_norm1(pool1_3x3_s2_b)
    conv2_3x3_reduce_b = conv2_3x3_reduce(pool1_norm1_b)
    conv2_3x3_b = conv2_3x3(conv2_3x3_reduce_b)
    conv2_norm2_b = conv2_norm2(conv2_3x3_b)
    conv2_zero_pad_b = conv2_zero_pad(conv2_norm2_b)
    pool2_helper_b = pool2_helper(conv2_zero_pad_b)
    pool2_3x3_s2_b = pool2_3x3_s2(pool2_helper_b)
    inception_3a_1x1_b = inception_3a_1x1(pool2_3x3_s2_b)
    inception_3a_3x3_reduce_b = inception_3a_3x3_reduce(pool2_3x3_s2_b)
    inception_3a_3x3_b = inception_3a_3x3(inception_3a_3x3_reduce_b)
    inception_3a_5x5_reduce_b = inception_3a_5x5_reduce(pool2_3x3_s2_b)
    inception_3a_5x5_b = inception_3a_5x5(inception_3a_5x5_reduce_b)
    inception_3a_pool_b = inception_3a_pool(pool2_3x3_s2_b)
    inception_3a_pool_proj_b = inception_3a_pool_proj(inception_3a_pool_b)

    inception_3a_output = Merge([inception_3a_1x1, inception_3a_3x3, inception_3a_5x5, inception_3a_pool_proj],
                                mode='concat', concat_axis=1, name='inception_3a/output')
    inception_3a_output_b = inception_3a_output(
        [inception_3a_1x1_b, inception_3a_3x3_b, inception_3a_5x5_b, inception_3a_pool_proj_b])
    inception_3b_1x1_b = inception_3b_1x1(inception_3a_output_b)
    inception_3b_3x3_reduce_b = inception_3b_3x3_reduce(inception_3a_output_b)
    inception_3b_3x3_b = inception_3b_3x3(inception_3b_3x3_reduce_b)
    inception_3b_5x5_reduce_b = inception_3b_5x5_reduce(inception_3a_output_b)
    inception_3b_5x5_b = inception_3b_5x5(inception_3b_5x5_reduce_b)
    inception_3b_pool_b = inception_3b_pool(inception_3a_output_b)
    inception_3b_pool_proj_b = inception_3b_pool_proj(inception_3b_pool_b)

    inception_3b_output = Merge([inception_3b_1x1, inception_3b_3x3, inception_3b_5x5, inception_3b_pool_proj],
                                mode='concat', concat_axis=1, name='inception_3b/output')
    inception_3b_output_b = inception_3b_output(
        [inception_3b_1x1_b, inception_3b_3x3_b, inception_3b_5x5_b, inception_3b_pool_proj_b])
    inception_3b_output_zero_pad_b = inception_3b_output_zero_pad(inception_3b_output_b)
    pool3_helper_b = pool3_helper(inception_3b_output_zero_pad_b)
    pool3_3x3_s2_b = pool3_3x3_s2(pool3_helper_b)
    inception_4a_1x1_b = inception_4a_1x1(pool3_3x3_s2_b)
    inception_4a_3x3_reduce_b = inception_4a_3x3_reduce(pool3_3x3_s2_b)
    inception_4a_3x3_b = inception_4a_3x3(inception_4a_3x3_reduce_b)
    inception_4a_5x5_reduce_b = inception_4a_5x5_reduce(pool3_3x3_s2_b)
    inception_4a_5x5_b = inception_4a_5x5(inception_4a_5x5_reduce_b)
    inception_4a_pool_b = inception_4a_pool(pool3_3x3_s2_b)
    inception_4a_pool_proj_b = inception_4a_pool_proj(inception_4a_pool_b)

    inception_4a_output = Merge([inception_4a_1x1, inception_4a_3x3, inception_4a_5x5, inception_4a_pool_proj],
                                mode='concat', concat_axis=1, name='inception_4a/output')
    inception_4a_output_b = inception_4a_output(
        [inception_4a_1x1_b, inception_4a_3x3_b, inception_4a_5x5_b, inception_4a_pool_proj_b])
    inception_4b_1x1_b = inception_4b_1x1(inception_4a_output_b)
    inception_4b_3x3_reduce_b = inception_4b_3x3_reduce(inception_4a_output_b)
    inception_4b_3x3_b = inception_4b_3x3(inception_4b_3x3_reduce_b)
    inception_4b_5x5_reduce_b = inception_4b_5x5_reduce(inception_4a_output_b)
    inception_4b_5x5_b = inception_4b_5x5(inception_4b_5x5_reduce_b)
    inception_4b_pool_b = inception_4b_pool(inception_4a_output_b)
    inception_4b_pool_proj_b = inception_4b_pool_proj(inception_4b_pool_b)

    inception_4b_output = Merge([inception_4b_1x1, inception_4b_3x3, inception_4b_5x5, inception_4b_pool_proj],
                                mode='concat', concat_axis=1, name='inception_4b_output')
    inception_4b_output_b = inception_4b_output(
        [inception_4b_1x1_b, inception_4b_3x3_b, inception_4b_5x5_b, inception_4b_pool_proj_b])
    inception_4c_1x1_b = inception_4c_1x1(inception_4b_output_b)
    inception_4c_3x3_reduce_b = inception_4c_3x3_reduce(inception_4b_output_b)
    inception_4c_3x3_b = inception_4c_3x3(inception_4c_3x3_reduce_b)
    inception_4c_5x5_reduce_b = inception_4c_5x5_reduce(inception_4b_output_b)
    inception_4c_5x5_b = inception_4c_5x5(inception_4c_5x5_reduce_b)
    inception_4c_pool_b = inception_4c_pool(inception_4b_output_b)
    inception_4c_pool_proj_b = inception_4c_pool_proj(inception_4c_pool_b)

    inception_4c_output = Merge([inception_4c_1x1, inception_4c_3x3, inception_4c_5x5, inception_4c_pool_proj],
                                mode='concat', concat_axis=1, name='inception_4c/output')
    inception_4c_output_b = inception_4c_output(
        [inception_4c_1x1_b, inception_4c_3x3_b, inception_4c_5x5_b, inception_4c_pool_proj_b])
    inception_4d_1x1_b = inception_4d_1x1(inception_4c_output_b)
    inception_4d_3x3_reduce_b = inception_4d_3x3_reduce(inception_4c_output_b)
    inception_4d_3x3_b = inception_4d_3x3(inception_4d_3x3_reduce_b)
    inception_4d_5x5_reduce_b = inception_4d_5x5_reduce(inception_4c_output_b)
    inception_4d_5x5_b = inception_4d_5x5(inception_4d_5x5_reduce_b)
    inception_4d_pool_b = inception_4d_pool(inception_4c_output_b)
    inception_4d_pool_proj_b = inception_4d_pool_proj(inception_4d_pool_b)

    inception_4d_output = Merge([inception_4d_1x1, inception_4d_3x3, inception_4d_5x5, inception_4d_pool_proj],
                                mode='concat', concat_axis=1, name='inception_4d/output')
    inception_4d_output_b = inception_4d_output(
        [inception_4d_1x1_b, inception_4d_3x3_b, inception_4d_5x5_b, inception_4d_pool_proj_b])
    inception_4e_1x1_b = inception_4e_1x1(inception_4d_output_b)
    inception_4e_3x3_reduce_b = inception_4e_3x3_reduce(inception_4d_output_b)
    inception_4e_3x3_b = inception_4e_3x3(inception_4e_3x3_reduce_b)
    inception_4e_5x5_reduce_b = inception_4e_5x5_reduce(inception_4d_output_b)
    inception_4e_5x5_b = inception_4e_5x5(inception_4e_5x5_reduce_b)
    inception_4e_pool_b = inception_4e_pool(inception_4d_output_b)
    inception_4e_pool_proj_b = inception_4e_pool_proj(inception_4e_pool_b)

    inception_4e_output = Merge([inception_4e_1x1, inception_4e_3x3, inception_4e_5x5, inception_4e_pool_proj],
                                mode='concat', concat_axis=1, name='inception_4e/output')
    inception_4e_output_b = inception_4e_output(
        [inception_4e_1x1_b, inception_4e_3x3_b, inception_4e_5x5_b, inception_4e_pool_proj_b])
    inception_4e_output_zero_pad_b = inception_4e_output_zero_pad(inception_4e_output_b)
    pool4_helper_b = pool4_helper(inception_4e_output_zero_pad_b)
    pool4_3x3_s2_b = pool4_3x3_s2(pool4_helper_b)
    inception_5a_1x1_b = inception_5a_1x1(pool4_3x3_s2_b)
    inception_5a_3x3_reduce_b = inception_5a_3x3_reduce(pool4_3x3_s2_b)
    inception_5a_3x3_b = inception_5a_3x3(inception_5a_3x3_reduce_b)
    inception_5a_5x5_reduce_b = inception_5a_5x5_reduce(pool4_3x3_s2_b)
    inception_5a_5x5_b = inception_5a_5x5(inception_5a_5x5_reduce_b)
    inception_5a_pool_b = inception_5a_pool(pool4_3x3_s2_b)
    inception_5a_pool_proj_b = inception_5a_pool_proj(inception_5a_pool_b)

    inception_5a_output = Merge([inception_5a_1x1, inception_5a_3x3, inception_5a_5x5, inception_5a_pool_proj],
                                mode='concat', concat_axis=1, name='inception_5a/output')
    inception_5a_output_b = inception_5a_output(
        [inception_5a_1x1_b, inception_5a_3x3_b, inception_5a_5x5_b, inception_5a_pool_proj_b])
    inception_5b_1x1_b = inception_5b_1x1(inception_5a_output_b)
    inception_5b_3x3_reduce_b = inception_5b_3x3_reduce(inception_5a_output_b)
    inception_5b_3x3_b = inception_5b_3x3(inception_5b_3x3_reduce_b)
    inception_5b_5x5_reduce_b = inception_5b_5x5_reduce(inception_5a_output_b)
    inception_5b_5x5_b = inception_5b_5x5(inception_5b_5x5_reduce_b)
    inception_5b_pool_b = inception_5b_pool(inception_5a_output_b)
    inception_5b_pool_proj_b = inception_5b_pool_proj(inception_5b_pool_b)

    inception_5b_output = Merge([inception_5b_1x1, inception_5b_3x3, inception_5b_5x5, inception_5b_pool_proj],
                                mode='concat', concat_axis=1, name='inception_5b/output')
    inception_5b_output_b = inception_5b_output(
        [inception_5b_1x1_b, inception_5b_3x3_b, inception_5b_5x5_b, inception_5b_pool_proj_b])
    pool5_7x7_s1_b = pool5_7x7_s1(inception_5b_output_b)
    loss3_flat_b = loss3_flat(pool5_7x7_s1_b)
    loss3_features_b = loss3_flat_b
    pool5_drop_7x7_s1_b = pool5_drop_7x7_s1(loss3_features_b)
    #    loss3_classifier_b = loss3_classifier(pool5_drop_7x7_s1_b)
    #    loss3_classifier_act_b = loss3_classifier_act(loss3_classifier_b)
    if num_score_layer == 2:
        score_1_b = score_1_layer(pool5_drop_7x7_s1_b)
        score_last_b = score_last_layer(score_1_b)
    elif num_score_layer == 3:
        score_1_b = score_1_layer(pool5_drop_7x7_s1_b)
        score_2_b = score_2_layer(score_1_b)
        score_last_b = score_last_layer(score_2_b)
    elif num_score_layer == 4:
        score_1_b = score_1_layer(pool5_drop_7x7_s1_b)
        score_2_b = score_2_layer(score_1_b)
        score_3_b = score_3_layer(score_2_b)
        score_last_b = score_last_layer(score_3_b)
    elif num_score_layer == 5:
        score_1_b = score_1_layer(pool5_drop_7x7_s1_b)
        score_2_b = score_2_layer(score_1_b)
        score_3_b = score_3_layer(score_2_b)
        score_4_b = score_4_layer(score_3_b)
        score_last_b = score_last_layer(score_4_b)
    elif num_score_layer == 6:
        score_1_b = score_1_layer(pool5_drop_7x7_s1_b)
        score_2_b = score_2_layer(score_1_b)
        score_3_b = score_3_layer(score_2_b)
        score_4_b = score_4_layer(score_3_b)
        score_5_b = score_5_layer(score_4_b)
        score_last_b = score_last_layer(score_5_b)
    else:
        score_last_b = score_last_layer(pool5_drop_7x7_s1_b)

    # ---------------------------------------------------------------------------------
    # input_a = Input(shape=(3, 224, 224))
    # Input 1 for google net.
    conv1_7x7_s2_a = conv1_7x7_s2(input_a)
    conv1_zero_pad_a = conv1_zero_pad(conv1_7x7_s2_a)
    pool1_helper_a = pool1_helper(conv1_zero_pad_a)
    pool1_3x3_s2_a = pool1_3x3_s2(pool1_helper_a)
    pool1_norm1_a = pool1_norm1(pool1_3x3_s2_a)
    conv2_3x3_reduce_a = conv2_3x3_reduce(pool1_norm1_a)
    conv2_3x3_a = conv2_3x3(conv2_3x3_reduce_a)
    conv2_norm2_a = conv2_norm2(conv2_3x3_a)
    conv2_zero_pad_a = conv2_zero_pad(conv2_norm2_a)
    pool2_helper_a = pool2_helper(conv2_zero_pad_a)
    pool2_3x3_s2_a = pool2_3x3_s2(pool2_helper_a)
    inception_3a_1x1_a = inception_3a_1x1(pool2_3x3_s2_a)
    inception_3a_3x3_reduce_a = inception_3a_3x3_reduce(pool2_3x3_s2_a)
    inception_3a_3x3_a = inception_3a_3x3(inception_3a_3x3_reduce_a)
    inception_3a_5x5_reduce_a = inception_3a_5x5_reduce(pool2_3x3_s2_a)
    inception_3a_5x5_a = inception_3a_5x5(inception_3a_5x5_reduce_a)
    inception_3a_pool_a = inception_3a_pool(pool2_3x3_s2_a)
    inception_3a_pool_proj_a = inception_3a_pool_proj(inception_3a_pool_a)
    inception_3a_output_a = inception_3a_output(
        [inception_3a_1x1_a, inception_3a_3x3_a, inception_3a_5x5_a, inception_3a_pool_proj_a])
    inception_3b_1x1_a = inception_3b_1x1(inception_3a_output_a)
    inception_3b_3x3_reduce_a = inception_3b_3x3_reduce(inception_3a_output_a)
    inception_3b_3x3_a = inception_3b_3x3(inception_3b_3x3_reduce_a)
    inception_3b_5x5_reduce_a = inception_3b_5x5_reduce(inception_3a_output_a)
    inception_3b_5x5_a = inception_3b_5x5(inception_3b_5x5_reduce_a)
    inception_3b_pool_a = inception_3b_pool(inception_3a_output_a)
    inception_3b_pool_proj_a = inception_3b_pool_proj(inception_3b_pool_a)
    inception_3b_output_a = inception_3b_output(
        [inception_3b_1x1_a, inception_3b_3x3_a, inception_3b_5x5_a, inception_3b_pool_proj_a])
    inception_3b_output_zero_pad_a = inception_3b_output_zero_pad(inception_3b_output_a)
    pool3_helper_a = pool3_helper(inception_3b_output_zero_pad_a)
    pool3_3x3_s2_a = pool3_3x3_s2(pool3_helper_a)
    inception_4a_1x1_a = inception_4a_1x1(pool3_3x3_s2_a)
    inception_4a_3x3_reduce_a = inception_4a_3x3_reduce(pool3_3x3_s2_a)
    inception_4a_3x3_a = inception_4a_3x3(inception_4a_3x3_reduce_a)
    inception_4a_5x5_reduce_a = inception_4a_5x5_reduce(pool3_3x3_s2_a)
    inception_4a_5x5_a = inception_4a_5x5(inception_4a_5x5_reduce_a)
    inception_4a_pool_a = inception_4a_pool(pool3_3x3_s2_a)
    inception_4a_pool_proj_a = inception_4a_pool_proj(inception_4a_pool_a)
    inception_4a_output_a = inception_4a_output(
        [inception_4a_1x1_a, inception_4a_3x3_a, inception_4a_5x5_a, inception_4a_pool_proj_a])
    inception_4b_1x1_a = inception_4b_1x1(inception_4a_output_a)
    inception_4b_3x3_reduce_a = inception_4b_3x3_reduce(inception_4a_output_a)
    inception_4b_3x3_a = inception_4b_3x3(inception_4b_3x3_reduce_a)
    inception_4b_5x5_reduce_a = inception_4b_5x5_reduce(inception_4a_output_a)
    inception_4b_5x5_a = inception_4b_5x5(inception_4b_5x5_reduce_a)
    inception_4b_pool_a = inception_4b_pool(inception_4a_output_a)
    inception_4b_pool_proj_a = inception_4b_pool_proj(inception_4b_pool_a)
    inception_4b_output_a = inception_4b_output(
        [inception_4b_1x1_a, inception_4b_3x3_a, inception_4b_5x5_a, inception_4b_pool_proj_a])
    inception_4c_1x1_a = inception_4c_1x1(inception_4b_output_a)
    inception_4c_3x3_reduce_a = inception_4c_3x3_reduce(inception_4b_output_a)
    inception_4c_3x3_a = inception_4c_3x3(inception_4c_3x3_reduce_a)
    inception_4c_5x5_reduce_a = inception_4c_5x5_reduce(inception_4b_output_a)
    inception_4c_5x5_a = inception_4c_5x5(inception_4c_5x5_reduce_a)
    inception_4c_pool_a = inception_4c_pool(inception_4b_output_a)
    inception_4c_pool_proj_a = inception_4c_pool_proj(inception_4c_pool_a)
    inception_4c_output_a = inception_4c_output(
        [inception_4c_1x1_a, inception_4c_3x3_a, inception_4c_5x5_a, inception_4c_pool_proj_a])
    inception_4d_1x1_a = inception_4d_1x1(inception_4c_output_a)
    inception_4d_3x3_reduce_a = inception_4d_3x3_reduce(inception_4c_output_a)
    inception_4d_3x3_a = inception_4d_3x3(inception_4d_3x3_reduce_a)
    inception_4d_5x5_reduce_a = inception_4d_5x5_reduce(inception_4c_output_a)
    inception_4d_5x5_a = inception_4d_5x5(inception_4d_5x5_reduce_a)
    inception_4d_pool_a = inception_4d_pool(inception_4c_output_a)
    inception_4d_pool_proj_a = inception_4d_pool_proj(inception_4d_pool_a)
    inception_4d_output_a = inception_4d_output(
        [inception_4d_1x1_a, inception_4d_3x3_a, inception_4d_5x5_a, inception_4d_pool_proj_a])
    inception_4e_1x1_a = inception_4e_1x1(inception_4d_output_a)
    inception_4e_3x3_reduce_a = inception_4e_3x3_reduce(inception_4d_output_a)
    inception_4e_3x3_a = inception_4e_3x3(inception_4e_3x3_reduce_a)
    inception_4e_5x5_reduce_a = inception_4e_5x5_reduce(inception_4d_output_a)
    inception_4e_5x5_a = inception_4e_5x5(inception_4e_5x5_reduce_a)
    inception_4e_pool_a = inception_4e_pool(inception_4d_output_a)
    inception_4e_pool_proj_a = inception_4e_pool_proj(inception_4e_pool_a)
    inception_4e_output_a = inception_4e_output(
        [inception_4e_1x1_a, inception_4e_3x3_a, inception_4e_5x5_a, inception_4e_pool_proj_a])
    inception_4e_output_zero_pad_a = inception_4e_output_zero_pad(inception_4e_output_a)
    pool4_helper_a = pool4_helper(inception_4e_output_zero_pad_a)
    pool4_3x3_s2_a = pool4_3x3_s2(pool4_helper_a)
    inception_5a_1x1_a = inception_5a_1x1(pool4_3x3_s2_a)
    inception_5a_3x3_reduce_a = inception_5a_3x3_reduce(pool4_3x3_s2_a)
    inception_5a_3x3_a = inception_5a_3x3(inception_5a_3x3_reduce_a)
    inception_5a_5x5_reduce_a = inception_5a_5x5_reduce(pool4_3x3_s2_a)
    inception_5a_5x5_a = inception_5a_5x5(inception_5a_5x5_reduce_a)
    inception_5a_pool_a = inception_5a_pool(pool4_3x3_s2_a)
    inception_5a_pool_proj_a = inception_5a_pool_proj(inception_5a_pool_a)
    inception_5a_output_a = inception_5a_output(
        [inception_5a_1x1_a, inception_5a_3x3_a, inception_5a_5x5_a, inception_5a_pool_proj_a])
    inception_5b_1x1_a = inception_5b_1x1(inception_5a_output_a)
    inception_5b_3x3_reduce_a = inception_5b_3x3_reduce(inception_5a_output_a)
    inception_5b_3x3_a = inception_5b_3x3(inception_5b_3x3_reduce_a)
    inception_5b_5x5_reduce_a = inception_5b_5x5_reduce(inception_5a_output_a)
    inception_5b_5x5_a = inception_5b_5x5(inception_5b_5x5_reduce_a)
    inception_5b_pool_a = inception_5b_pool(inception_5a_output_a)
    inception_5b_pool_proj_a = inception_5b_pool_proj(inception_5b_pool_a)
    inception_5b_output_a = inception_5b_output(
        [inception_5b_1x1_a, inception_5b_3x3_a, inception_5b_5x5_a, inception_5b_pool_proj_a])
    pool5_7x7_s1_a = pool5_7x7_s1(inception_5b_output_a)
    loss3_flat_a = loss3_flat(pool5_7x7_s1_a)
    loss3_features_a = loss3_flat_a
    pool5_drop_7x7_s1_a = pool5_drop_7x7_s1(loss3_features_a)
    #    loss3_classifier_a = loss3_classifier(pool5_drop_7x7_s1_a)
    #    loss3_classifier_act_a = loss3_classifier_act(loss3_classifier_a)
    if num_score_layer == 2:
        score_1_a = score_1_layer(pool5_drop_7x7_s1_a)
        score_last_a = score_last_layer(score_1_a)
    elif num_score_layer == 3:
        score_1_a = score_1_layer(pool5_drop_7x7_s1_a)
        score_2_a = score_2_layer(score_1_a)
        score_last_a = score_last_layer(score_2_a)
    elif num_score_layer == 4:
        score_1_a = score_1_layer(pool5_drop_7x7_s1_a)
        score_2_a = score_2_layer(score_1_a)
        score_3_a = score_3_layer(score_2_a)
        score_last_a = score_last_layer(score_3_a)
    elif num_score_layer == 5:
        score_1_a = score_1_layer(pool5_drop_7x7_s1_a)
        score_2_a = score_2_layer(score_1_a)
        score_3_a = score_3_layer(score_2_a)
        score_4_a = score_4_layer(score_3_a)
        score_last_a = score_last_layer(score_4_a)
    elif num_score_layer == 6:
        score_1_a = score_1_layer(pool5_drop_7x7_s1_a)
        score_2_a = score_2_layer(score_1_a)
        score_3_a = score_3_layer(score_2_a)
        score_4_a = score_4_layer(score_3_a)
        score_5_a = score_5_layer(score_4_a)
        score_last_a = score_last_layer(score_5_a)
    else:
        score_last_a = score_last_layer(pool5_drop_7x7_s1_a)

    # si-sj
    distance = Lambda(BTPred, output_shape=(1,))([score_last_a, score_last_b])
    comp_net_functional = Model(inputs=[input_a, input_b], outputs=[distance])
    return comp_net_functional

if __name__ == "__main__":
    num_of_classes = 3
    num_of_features = 1024
    score_layer = 4

    input_a = Input(shape=(3, 224, 224))
    input_b = Input(shape=(3, 224, 224))
    base_network = create_base_network(input_a, input_b, no_classes=num_of_classes, no_features=num_of_features, num_score_layer=score_layer)
    base_network.summary()
    plot_model(base_network, to_file='model.png')