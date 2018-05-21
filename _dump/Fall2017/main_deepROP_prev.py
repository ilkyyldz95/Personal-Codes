from deepROP import deep_ROP
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'James NN',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', type=str, default='reg', choices=['train', 'test'],
                        help='choose training the neural network or evaluating it.')
    parser.add_argument('k',type=int, help='k-th fold to be excluded in training or k-th fold to be tests.')
    args = parser.parse_args()

    #######################################################################################################
    # Setting parameters
    #######################################################################################################
    partition_file = './6000Partitions.p'
    img_folder='./preprocessed_JamesCode/'
    init_weight = './googlenet_weights.h5'
    epochs = 100
    learning_rate = 1e-4
    batch_size = 32
    loss = 'categorical_crossentropy'
    save_model_name = './deep_rop_bal_d05_cross_lr'+str(learning_rate)+'_epochs'+str(epochs)+'_k_'+str(args.k)+'.h5'
    #######################################################################################################
    if args.mode == 'train':
        deep_ROP_net = deep_ROP(partition_file, img_folder)
        deep_ROP_net.train(args.k, init_weight=init_weight, save_model_name=save_model_name,
              epochs=epochs, learning_rate=learning_rate, batch_size=32, loss=loss,
              num_of_classes = 3, num_of_features = 1024, drop_rate=0.5)
    elif args.mode == 'test':
        model_name = './deep_rop_bal_d05_cross_lr' + str(learning_rate) + '_epochs' + str(epochs)+'_k_'+str(args.k)+ '.h5'
        deep_ROP_net = deep_ROP(partition_file, img_folder)
        deep_ROP_net.test(args.k,model_file=model_name,num_of_classes = 3)
    else:
        print('mode must be either train or test')