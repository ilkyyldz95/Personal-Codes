from deepROP import deep_ROP
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='James NN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', type=str, default='reg', choices=['train', 'test'],
                        help='choose training the neural network or evaluating it.')
    parser.add_argument('k', type=int, help='k-th fold to be excluded in training or k-th fold to be tests.')
    parser.add_argument('no_im', type=int, default=12, help='num of unique images trained')
    args = parser.parse_args()
    #######################################################################################################
    # Setting parameters
    #######################################################################################################
    num_unique_images = int(args.no_im)
    save_model_name = 'RSD_Fold_' + str(args.k) + '_no_unique_' + str(num_unique_images) + '.h5'
    test_100 = True #validation on 100 if True, test on 5000 if False. kthFold is 5 for test
    balance = False
    #####################################
    partition_file_6000 = './6000Partitions.p'
    partition_file_100 = './Partitions.p'
    img_folder_6000 = './preprocessed_JamesCode/'
    img_folder_100 = './preprocessed/All/'
    init_weight = './googlenet_weights.h5'
    loss = 'categorical_crossentropy'
    epochs = 50
    learning_rate = 1e-4
    num_of_classes = 3
    #######################################################################################################
    if args.mode == 'train':
        deep_ROP_net = deep_ROP(partition_file_100=partition_file_100, img_folder_100=img_folder_100,
                        partition_file_6000=partition_file_6000, img_folder_6000=img_folder_6000)
        deep_ROP_net.train(args.k, init_weight=init_weight, save_model_name=save_model_name,
                        epochs=epochs, learning_rate=learning_rate, num_unique_images=num_unique_images,
                        batch_size=32, loss=loss, num_of_classes=num_of_classes, num_of_features=1024,
                        balance=balance)
    elif args.mode == 'test':
        deep_ROP_net = deep_ROP(partition_file_100=partition_file_100, img_folder_100=img_folder_100,
                        partition_file_6000=partition_file_6000, img_folder_6000=img_folder_6000)
        deep_ROP_net.test(args.k, save_model_name, learning_rate=learning_rate, loss=loss,
                        num_of_classes=num_of_classes, num_of_features=1024, test_100=test_100,
                        num_unique_images=num_unique_images)
    else:
        print('mode must be either train or test')
