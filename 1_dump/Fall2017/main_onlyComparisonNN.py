import argparse
from onlyComparisonNN import *

if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Only Comparison NN',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', type=str, default='reg', choices=['train', 'test'],
                        help='choose training the neural network or evaluating it.')
    parser.add_argument('k',type=int, help='k-th fold to be excluded in training or k-th fold to be tests.')
    parser.add_argument('layers', type=int, help='num of layers to calculate score')
    parser.add_argument('loss', type=str, help='comparison loss')
    parser.add_argument('no_im', type=int, help='num of unique images trained')
    args = parser.parse_args()
    #######################################################################################################
    # Setting parameters
    #######################################################################################################
    kthFold = int(args.k)
    score_layer = int(args.layers)
    if args.loss == "Thurstone":
        comp_loss = ThurstoneLoss
    elif args.loss == "diff":
        comp_loss = diffLoss
    else:
        comp_loss = BTLoss
    num_unique_images = int(args.no_im)
    save_model_name = 'Comp_Only_Fold_' + str(kthFold) + '_layers_' + str(score_layer) + \
                      '_loss_' + str(args.loss) + '.h5'
    #####################################
    partition_file_6000 = './6000Partitions.p'
    partition_file_100 = './Partitions.p'
    img_folder_6000 = './preprocessed_JamesCode/'
    img_folder_100 = './preprocessed/All/'
    init_weight = './googlenet_weights.h5'
    epochs = 50
    learning_rate = 1e-4
    num_of_classes = 3
    #######################################################################################################
    if args.mode == 'train':
        comp_net = onlyComparisonNN(partition_file_100=partition_file_100, img_folder_100=img_folder_100,
                                partition_file_6000=partition_file_6000, img_folder_6000=img_folder_6000)
        comp_net.train(kthFold, init_weight=init_weight, save_model_name=save_model_name,
              epochs=epochs, learning_rate=learning_rate, num_unique_images=num_unique_images, batch_size=24,
              comp_loss=comp_loss, num_of_classes=num_of_classes, num_of_features=1024, score_layer=score_layer)
    elif args.mode == 'test':
        comp_net = onlyComparisonNN(partition_file_100=partition_file_100, img_folder_100=img_folder_100,
                                partition_file_6000=partition_file_6000, img_folder_6000=img_folder_6000)
        comp_net.test(kthFold, save_model_name, learning_rate=learning_rate, comp_loss=comp_loss,
              num_of_classes=num_of_classes, num_of_features=1024, score_layer=score_layer, test_100=True,
              num_unique_images=num_unique_images)
    else:
        print('mode must be either train or test')
