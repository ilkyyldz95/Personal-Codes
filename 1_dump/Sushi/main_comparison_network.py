import argparse
from comparison_network import *
from keras import optimizers

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='NN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', type=str, default='reg', choices=['train', 'test'],
                        help='choose training the neural network or evaluating it.')
    parser.add_argument('no_user', type=int)
    parser.add_argument('samp_per_user', type=int)
    args = parser.parse_args()
    #######################################################################################################
    # Setting parameters
    #######################################################################################################
    kthFold = 0
    save_model_name = 'Comp_model_no_user_' + str(args.no_user) + '_samp_per_user_' + str(args.samp_per_user)
    no_of_labelers = args.no_user
    sample_per_user = args.samp_per_user
    no_of_base_layers = 2
    max_no_of_nodes = 30
    epochs = 100
    optimizer = optimizers.RMSprop()
    batch_size = 128
    loss = BTLoss
    #######################################################################################################
    if args.mode == 'train':
        comp_net = comparison_network()
        comp_net.train(kthFold, save_model_name=save_model_name,
              no_of_labelers = no_of_labelers, sample_per_user=sample_per_user,
              no_of_base_layers=no_of_base_layers, max_no_of_nodes=max_no_of_nodes,
              epochs=epochs, optimizer=optimizer, batch_size=batch_size, loss=loss)
    elif args.mode == 'test':
        comp_net = comparison_network()
        comp_net.test(kthFold, save_model_name,
                      no_of_labelers=no_of_labelers, sample_per_user=sample_per_user,
                      no_of_base_layers=no_of_base_layers, max_no_of_nodes=max_no_of_nodes,
                      optimizer=optimizer, loss=loss)
    else:
        print('mode must be either train or test')
