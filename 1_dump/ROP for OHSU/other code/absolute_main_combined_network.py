import argparse

from absolute_combined_deep_ROP import *

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='NN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', type=str, default='reg', choices=['train', 'test'],
                        help='choose training the neural network or evaluating it.')
    parser.add_argument('alpha', type=float)
    parser.add_argument('no_im', type=int)
    parser.add_argument('reg_param', type=float)
    parser.add_argument('lr', type=float)
    args = parser.parse_args()
    #######################################################################################################
    # Setting parameters
    #######################################################################################################
    kthFold = 3  # test fold:3, validation fold:0, train with 1,2,4
    reg_param = args.reg_param
    learning_rate = args.lr
    alpha = args.alpha
    num_unique_images = args.no_im

    save_model_name = 'model_alpha_' + str(alpha) + '_no_im_' + str(num_unique_images) + \
                      '_lambda_' + str(reg_param) + '_lr_' + str(learning_rate)
    ################# fixed
    abs_test_thr = 'plus'
    no_of_classes = 3
    no_of_score_layers = 1
    # max_no_of_nodes = 128
    abs_loss = scaledCrossEntropy
    comp_loss = scaledBTLoss
    epochs = 50
    batch_size = 24
    balance = False
    #######################################################################################################
    if args.mode == 'train':
        combined_net = combined_deep_ROP()
        combined_net.train(kthFold, save_model_name=save_model_name,
                           reg_param=reg_param, no_of_classes=no_of_classes, no_of_score_layers=no_of_score_layers,
                           abs_loss=abs_loss, comp_loss=comp_loss, alpha=alpha, learning_rate=learning_rate,
                           epochs=epochs, num_unique_images=num_unique_images, batch_size=batch_size, balance=balance)
    elif args.mode == 'test':
        combined_net = combined_deep_ROP()
        combined_net.test(kthFold, save_model_name,
                          reg_param=reg_param, no_of_classes=no_of_classes, no_of_score_layers=no_of_score_layers,
                          abs_loss=abs_loss, comp_loss=comp_loss, alpha=alpha, learning_rate=learning_rate,
                          num_unique_images=num_unique_images, balance=balance, abs_test_thr=abs_test_thr)
    else:
        print('mode must be either train or test')
