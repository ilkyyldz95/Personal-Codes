import argparse

from combined_fac import *

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='NN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', type=str, default='reg', choices=['train', 'test'],
                        help='choose training the neural network or evaluating it.')
    parser.add_argument('reg_param', type=float)
    parser.add_argument('lr', type=float)
    args = parser.parse_args()
    #######################################################################################################
    # Setting parameters
    #######################################################################################################
    epochs = 75
    batch_size = 10
    reg_param = args.reg_param
    learning_rate = args.lr
    no_of_fused_features = 256
    ###
    input_shape = (3, 224, 224)
    dir = "./IMAGE_QUALITY_DATA"
    save_model_name = 'model_lambda_' + str(reg_param) + '_lr_' + str(learning_rate)
    #######################################################################################################
    if args.mode == 'train':
        combined_net = combined_fac(input_shape=input_shape, dir=dir)
        combined_net.train(save_model_name=save_model_name,
              reg_param=reg_param, no_of_fused_features=no_of_fused_features, learning_rate=learning_rate,
              epochs=epochs, batch_size=batch_size)
    elif args.mode == 'test':
        combined_net = combined_fac(input_shape=input_shape, dir=dir)
        combined_net.test(save_model_name, reg_param=reg_param, no_of_fused_features=no_of_fused_features,
              learning_rate=learning_rate)
    else:
        print('mode must be either train or test')
