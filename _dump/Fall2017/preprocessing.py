import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from scipy.ndimage import rotate


def biLabels(labels):
    """
    This function will binarized labels.
    There are C classes {1,2,3,4,...,c} in the labels, the output would be c dimensional vector.
    Input:
        - labels: (N,) np array. The element value indicates the class index.
    Output:
        - biLabels: (N, C) array. Each row has and only has a 1, and the other elements are all zeros.
        - C: integer. The number of classes in the data.
    Example:
        The input labels = np.array([1,2,2,1,3])
        The binaried labels are np.array([[1,0,0],[0,1,0],[0,1,0],[1,0,0],[0,0,1]])
    """
    N = labels.shape[0]
    labels.astype(np.int)
    C = len(np.unique(labels))
    binarized = np.zeros((N, C))
    binarized[np.arange(N).astype(np.int), labels.astype(np.int) - 1] = 1
    return binarized, C

def rotate_data(img_ori, label_bi):
    """
    rotate images to 90, 180, 270 , 360 degrees. A single image will have 4 variants.
    img_train_ori: image array (N,3,x,y). N is the number of images in the array.
    label_train_bi: label array (N, c) c is the number of classes of the data.
    return:
    img_rotate: rotated image array (4*N, 3, x, y).
    label_rotated : label array (4*N, c) c is the number of classes of the data.
    """
    img_rotate_90 = rotate(img_ori, 90, axes=(2, 3))
    img_rotate_90_ori = np.concatenate((img_ori, img_rotate_90), axis=0)
    img_rotate_180 = rotate(img_rotate_90_ori, 180, axes=(2, 3))
    img_rotated = np.concatenate((img_rotate_90_ori, img_rotate_180), axis=0)
    label_rotated = np.tile(label_bi, (4, 1))
    return img_rotated, label_rotated


def flip_data(img_ori, label_bi):
    """
    flip images. A single image will have 2 variants.
    img_train_ori: image array (N,3,x,y). N is the number of images in the array.
    label_train_bi: label array (N, c) c is the number of classes of the data.
    return:
    img_flipped: flip image array (2*N, 3, x, y).
    label_flipped : label array (2*N, c) c is the number of classes of the data.
    """
    img_flip = np.flipud(img_ori)
    img_flipped = np.concatenate((img_ori, np.flipud(img_flip)), axis=0)
    label_flipped = np.tile(label_bi, (2, 1))
    return img_flipped, label_flipped


def augment_data(img_ori, label_bi, rotation=True, flip=True):
    """
    Image augmentation with roattions and flips.
    img_train_ori: image array (N,3,x,y). N is the number of images in the array.
    label_train_bi: label array (N, c) c is the number of classes of the data.
    return:
    img_augmented (n*N, 3, x, y)
    label_augmented  (n*N, 3, x, y)
    """
    if rotation:
        img_rotated, label_rotated = rotate_data(img_ori, label_bi)
        if flip:
            img_rotated_flipped, label_rotated_flipped = flip_data(img_rotated, label_rotated)
            return img_rotated_flipped, label_rotated_flipped
        else:
            return img_rotated, label_rotated
    else:
        if flip:
            img_flipped, label_flipped = flip_data(img_ori, label_bi)
            return img_flipped, label_flipped
        else:
            return img_ori, label_bi

def balance_class_data(img,label_bi, num_unique_images=12):
    """
    balance the training data which contains the exact the number of images from each class.
    img (N, 3, x, y) image array. N is the number of images
    label_bi (N,c) numpy array. c is the number of classes.
    return:
    img_balanced: (m, 3, x, y) image array. m is the number of total balanced images.
    label_bi_balanced: (m,c) array.
    """
    c = label_bi.shape[1]
    ind_list = []
    len_ind_list = []
    for c_iter in range(c):
        ind_list.append(np.where(label_bi[:, c_iter] == 1)[0])
        len_ind_list.append(len(ind_list[c_iter]))
    min_unique_img_class = np.floor(1. * num_unique_images / 3)
    min_size_class = min(len_ind_list)
    if num_unique_images !=0:
        if min_unique_img_class>=min_size_class:
            raise ValueError("The maximum number of images for single class is: "+str(min_size_class))
        else:
            min_size_class=int(min_unique_img_class)
    np.random.seed(1)
    ind_balanced_list = []
    for c_balance_iter in range(c):
        ind_balanced_list.append(ind_list[c_balance_iter][np.random.choice(len_ind_list[c_balance_iter],min_size_class)])
    ind_balanced = np.concatenate(ind_balanced_list)
    img_balanced = img[ind_balanced,:,:,:]
    label_bi_balanced = label_bi[ind_balanced,:]
    print('Data Balance Complete.')
    return img_balanced, label_bi_balanced,ind_balanced

def drop_class_data(img,label_bi,drop_rate):
    """
    use part of balanced the training data which contains the exact the number of images from each class.
    img (N, 3, x, y) image array. N is the number of images
    label_bi (N,c) numpy array. c is the number of classes.
    return:
    img_balanced: (m, 3, x, y) image array. m is the number of total dropped balanced images.
    label_bi_balanced: (m,c) array.
    """
    c = label_bi.shape[1]
    ind_list = []
    len_ind_list = []
    for c_iter in range(c):
        ind_list.append(np.where(label_bi[:, c_iter] == 1)[0])
        len_ind_list.append(len(ind_list[c_iter]))
    min_size_class = min(len_ind_list)
    drop_min_size_class = int(np.around(1.*min_size_class*drop_rate))
    np.random.seed(1)
    ind_balanced_list = []
    for c_balance_iter in range(c):
        ind_balanced_list.append(ind_list[c_balance_iter][np.random.choice(len_ind_list[c_balance_iter],drop_min_size_class)])
    ind_balanced = np.concatenate(ind_balanced_list)
    img_balanced = img[ind_balanced,:,:,:]
    label_bi_balanced = label_bi[ind_balanced,:]
    return img_balanced, label_bi_balanced,ind_balanced_list

def obtain_comparison_indices(img_index_train, imgNames100, part_cmp_index, cmp_pair_names_experts):
    """
    This function is to get the training comparison data from the given indices of 1st 100 images.
    Input:
        - img_index_train: The index of 1st 100 images.
    Output:
        - cmp_indices:
        - cmp_labels:
    """
    train_rsd_img_names = [imgNames100[img_index+1] for img_index in list(img_index_train)]
    cmp_indices = []
    for i in part_cmp_index:
        img_name_i, img_name_j = cmp_pair_names_experts[0][i][0], cmp_pair_names_experts[0][i][1]
        if img_name_i in train_rsd_img_names and img_name_j in train_rsd_img_names:
            cmp_indices.append(i)
    return cmp_indices