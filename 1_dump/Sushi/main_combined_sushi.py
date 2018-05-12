import argparse
from combined_sushi import *
from keras import optimizers

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='NN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', type=str, default='reg', choices=['train', 'test'],
                        help='choose training the neural network or evaluating it.')
    parser.add_argument('train', type=str)
    parser.add_argument('sim_thr', type=float)
    parser.add_argument('samp_per_user', type=int)
    args = parser.parse_args()
    #######################################################################################################
    # Setting parameters
    #######################################################################################################
    kthFold = 0
    save_model_name = 'combined_model_sim_thr_' + str(args.sim_thr) + '_samp_per_user_' + str(args.samp_per_user)
    train_set = args.train
    # no_of_labelers = args.no_user
    user_sim_thr = args.sim_thr
    sample_per_user = args.samp_per_user
    no_of_base_layers = 2
    max_no_of_nodes = 30
    epochs = 100
    optimizer = optimizers.RMSprop()
    batch_size = 128
    comp_loss = scaledBTLoss
    abs_loss = scaledCrossEntropy
    #######################################################################################################
    if args.mode == 'train':
        combined_net = combined_network()
        combined_net.train(kthFold, save_model_name=save_model_name, train_set=train_set, user_sim_thr=user_sim_thr,
              sample_per_user=sample_per_user, no_of_base_layers=no_of_base_layers, max_no_of_nodes=max_no_of_nodes,
              epochs=epochs, optimizer=optimizer, batch_size=batch_size, comp_loss=comp_loss, abs_loss=abs_loss)
    elif args.mode == 'test':
        combined_net = combined_network()
        combined_net.test(kthFold, save_model_name, train_set=train_set, user_sim_thr=user_sim_thr,
              sample_per_user=sample_per_user, no_of_base_layers=no_of_base_layers, max_no_of_nodes=max_no_of_nodes,
              optimizer=optimizer, comp_loss=comp_loss, abs_loss=abs_loss)
    else:
        print('mode must be either train or test')
