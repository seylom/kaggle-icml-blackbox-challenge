from pylearn2.scripts.icml_2013_wrepl.black_box.black_box_dataset import BlackBoxDataset

MAX_EPOCHS_UNSUPERVISED = 50
MAX_EPOCHS_SUPERVISED = 200

from pylearn2.corruption import BinomialCorruptor
from pylearn2.corruption import GaussianCorruptor
from pylearn2.costs.mlp import Default
from pylearn2.models.autoencoder import Autoencoder, DenoisingAutoencoder
from pylearn2.models.rbm import GaussianBinaryRBM
from pylearn2.models.softmax_regression import SoftmaxRegression
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.training_algorithms.sgd import LinearDecayOverEpoch
from pylearn2.costs.autoencoder import MeanBinaryCrossEntropy
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.energy_functions.rbm_energy import GRBM_Type_1
from pylearn2.base import StackedBlocks
from pylearn2.datasets.transformer_dataset import TransformerDataset
from pylearn2.costs.ebm_estimation import SMD
from pylearn2.training_algorithms.sgd import MonitorBasedLRAdjuster
from pylearn2.termination_criteria import MonitorBased
from pylearn2.train import Train
from pylearn2.models.mlp import Softmax
from pylearn2.models.mlp import MLP
from pylearn2.models.mlp import PretrainedLayer
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.datasets.preprocessing import Standardize
from sklearn import preprocessing

import pylearn2.utils.serial as serial
import numpy as np
from pylearn2.utils.string_utils import preprocess

import os

import numpy.random

def get_dataset_icml():

    base_path = '${PYLEARN2_DATA_PATH}/icml_2013_black_box'
    
    min_max_scaler = preprocessing.MinMaxScaler()
    
    #process= Standardize()
       
    trainset = BlackBoxDataset(which_set = 'train',start = 0, stop = 900)
    validset = BlackBoxDataset(which_set = 'train',start = 900, stop = 1000)
    extraset = BlackBoxDataset(which_set = 'extra',start = 0, stop = 1000) #trainset
    testset =  BlackBoxDataset(which_set = 'test')
    
    extraset.X = min_max_scaler.fit_transform(extraset.X)
    trainset.X = min_max_scaler.transform(trainset.X)
    validset.X = min_max_scaler.transform(validset.X)
    
#    process.apply(extraset,can_fit=True)
#    process.apply(trainset)
#    process.apply(validset)
#    process.apply(testset)

    return trainset, validset, testset, extraset

def get_grbm(structure):
    n_input, n_output = structure
    config = {
        'nvis': n_input,
        'nhid': n_output,
        "irange" : 0.05,
        "energy_function_class" : GRBM_Type_1,
        "learn_sigma" : True,
        "init_sigma" : .4,
        "init_bias_hid" : -2.,
        "mean_vis" : False,
        "sigma_lr_scale" : 1e-3
        }

    return GaussianBinaryRBM(**config)


def get_denoising_autoencoder(structure,corr_val):
    n_input, n_output = structure
    #corruptor = BinomialCorruptor(corruption_level=corr_val)
    corruptor =  GaussianCorruptor(stdev=0.25)
    config = {
        'corruptor': corruptor,
        'nhid': n_output,
        'nvis': n_input,
        'tied_weights': True,
        'act_enc': 'sigmoid',
        'act_dec': 'sigmoid',
        'irange': 4*np.sqrt(6. / (n_input + n_output)),
    }
    return DenoisingAutoencoder(**config)

def get_logistic_regressor(structure):
    n_input, n_output = structure

    layer = SoftmaxRegression(n_classes=n_output, 
                    irange= 0.05, 
                    nvis=n_input)

    return layer

def get_layer_trainer_logistic(layer, trainset,validset):
    # configs on sgd

    config = {'learning_rate': 0.1,
              'cost' : Default(),
              'batch_size': 150,
              'monitoring_dataset': validset,
              'termination_criterion': MonitorBased(channel_name='y_misclass',N=10,prop_decrease=0),
              'update_callbacks': None
              }

    train_algo = SGD(**config)
    model = layer
    return Train(model = model,
            dataset = trainset,
            algorithm = train_algo,
            extensions = None)

def get_layer_trainer_sgd_autoencoder(layer, trainset,savepkl):
    # configs on sgd
    train_algo = SGD(
              learning_rate = 0.05,
              cost = MeanSquaredReconstructionError(), #MeanBinaryCrossEntropy(), #MeanSquaredReconstructionError() 
              batch_size =  100,
              monitoring_batches = 5,
              monitoring_dataset =  trainset,
              termination_criterion = EpochCounter(max_epochs=MAX_EPOCHS_UNSUPERVISED),
              update_callbacks =  None
              )

    model = layer
    extensions = None
    return Train(model = model,
            algorithm = train_algo,
            extensions = extensions,
            dataset = trainset,
            save_path = savepkl,
            save_freq = 100)
    
def get_layer_trainer_sgd_rbm(layer, trainset):
    train_algo = SGD(
        learning_rate = 1e-1,
        batch_size =  5,
        #"batches_per_iter" : 2000,
        monitoring_batches =  20,
        monitoring_dataset =  trainset,
        cost = SMD(corruptor = GaussianCorruptor(stdev=0.4)),
        termination_criterion =  EpochCounter(max_epochs=MAX_EPOCHS_UNSUPERVISED),
        )
    model = layer
    extensions = [MonitorBasedLRAdjuster()]
    return Train(model = model, algorithm = train_algo,
                 #save_path='grbm.pkl',save_freq=1,
                 extensions = extensions, dataset = trainset)
    
def get_layer_MLP(layers,trainset,validset):
    
    #processor = Standardize();
    
#    trainset = BlackBoxDataset( which_set = 'train',
#                                start = 0,
#                                stop = 900,
#                                preprocessor = Standardize(),
#                                fit_preprocessor = True,
#                                fit_test_preprocessor = True,
#                                )
#    
#    validset = BlackBoxDataset( which_set = 'train',
#                                start = 900,
#                                stop = 1000 ,
#                                preprocessor = Standardize(),
#                                fit_preprocessor = True,
#                                fit_test_preprocessor = False,
#                                )
    
    dropCfg = { 'input_include_probs': { 'h0' : .8 } ,
                'input_scales': { 'h0': 1.}
              }
    
    config = { 'learning_rate': .1,
                'init_momentum': .00,
                'cost' :  Default(), #Dropout(**dropCfg),
                'monitoring_dataset':  { 'train' : trainset,
                                         'valid' : validset
                                        },
                'termination_criterion': MonitorBased(channel_name='valid_y_misclass',N=10,prop_decrease=0),
                'update_callbacks': None
              }
     
#    configCfg0 = {'layer_name' : 'h0',
#                'dim' : 1875,
#                'irange' : .05,
#                # Rather than using weight decay, we constrain the norms of the weight vectors
#                 'max_col_norm' : 1.}
#    
#    configCfg1 = {'layer_name' : 'h1',
#                'dim' : 1875,
#                'irange' : .05,
#                # Rather than using weight decay, we constrain the norms of the weight vectors
#                 'max_col_norm' : 1.}
    
    sftmaxCfg = {
                'layer_name': 'y',
                'init_bias_target_marginals': trainset,
                # Initialize the weights to all 0s
                'irange': .0,
                'n_classes': 9
            }
    
    layers.append(Softmax(**sftmaxCfg)) 

    train_algo = SGD(**config)
    model = MLP(batch_size=100,layers=layers,nvis=1875)
    return Train(model = model,
            dataset = trainset,
            algorithm = train_algo,
            extensions = None, #[LinearDecayOverEpoch(start= 5, saturate= 100, decay_factor= .01)], 
            save_path = "sae_best_model.pkl",
            save_freq = 100)

def main():

    trainset, validset, testset, extraset = get_dataset_icml()
    #trainset,testset = get_dataset_mnist()
    
    design_matrix = trainset.get_design_matrix()
    n_input = design_matrix.shape[1]
    
    n_output = 9 #10

    # build layers
    layers = []
    structure = [[n_input, 2000], [2000,2000],[2000,2000], [2000, n_output]]
    
    #layers.append(get_grbm(structure[0]))
    # layer 0: denoising AE
    layers.append(get_denoising_autoencoder(structure[0],0.25))
    # layer 1: denoising AE
    layers.append(get_denoising_autoencoder(structure[1],0.25))
     # layer 1: denoising AE
    layers.append(get_denoising_autoencoder(structure[2],0.25))
    # layer 2: logistic regression used in supervised training
    layers.append(get_logistic_regressor(structure[3]))


    #construct training sets for different layers
    traindata = [ extraset ,
                TransformerDataset( raw = extraset, transformer = layers[0] ),
                TransformerDataset( raw = extraset, transformer = StackedBlocks( layers[0:2] )),
                TransformerDataset( raw = trainset, transformer = StackedBlocks( layers[0:3] )) ]
    
    #valid =  TransformerDataset( raw = validset, transformer = StackedBlocks( layers[0:2] ))
    
    #valid = trainset

    # construct layer trainers
    layer_trainers = []
    #layer_trainers.append(get_layer_trainer_sgd_rbm(layers[0], trainset[0]))
    layer_trainers.append(get_layer_trainer_sgd_autoencoder(layers[0], traindata[0],'ae1.pkl'))
    layer_trainers.append(get_layer_trainer_sgd_autoencoder(layers[1], traindata[1],'ae2.pkl'))
    layer_trainers.append(get_layer_trainer_sgd_autoencoder(layers[2], traindata[2],'ae3.pkl'))
    #layer_trainers.append(get_layer_trainer_logistic(layers[3], traindata[3],traindata[3]))

#    #unsupervised pretraining
#    for i, layer_trainer in enumerate(layer_trainers[0:3]):
#        print '-----------------------------------'
#        print ' Unsupervised training (pretraining) layer %d, %s'%(i, layers[i].__class__)
#        print '-----------------------------------'
#        layer_trainer.main_loop()
#
#
#    print '\n'
#    print '------------------------------------------------------'
#    print ' Unsupervised training done! Start supervised training (fine-tuning)...'
#    print '------------------------------------------------------'
#    print '\n'
    
    mlp_layers = []
    mlp_layers.append(PretrainedLayer(layer_name = 'h0', layer_content = serial.load('ae1.pkl')))
    mlp_layers.append(PretrainedLayer(layer_name = 'h1', layer_content = serial.load('ae2.pkl')))
    mlp_layers.append(PretrainedLayer(layer_name = 'h2', layer_content = serial.load('ae3.pkl')))
#
    #supervised training
    #layer_trainers[-1].main_loop()
    mlp_model = get_layer_MLP(mlp_layers,trainset,validset)
    mlp_model.main_loop()
#
#    layer_trainers[-1].main_loop()


if __name__ == '__main__':
    main()
