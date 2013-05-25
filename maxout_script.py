"""
See readme.txt

A small example of how to glue shining features of pylearn2 together
to train models layer by layer.
"""
from pylearn2.scripts.icml_2013_wrepl.black_box.black_box_dataset import BlackBoxDataset

MAX_EPOCHS_UNSUPERVISED = 50
MAX_EPOCHS_SUPERVISED = 2

from pylearn2.corruption import BinomialCorruptor
from pylearn2.corruption import GaussianCorruptor
from pylearn2.costs.mlp import Default
from pylearn2.models.autoencoder import Autoencoder, DenoisingAutoencoder
from pylearn2.models.maxout import Maxout
from pylearn2.models.softmax_regression import SoftmaxRegression
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets import cifar10
from pylearn2.datasets import mnist
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
from pylearn2.models.mlp import RectifiedLinear
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.datasets.preprocessing import Standardize

import pylearn2.utils.serial as serial
import numpy as np
from pylearn2.utils.string_utils import preprocess

import os

import numpy.random

def get_dataset_icml():

    base_path = '${PYLEARN2_DATA_PATH}/icml_2013_black_box'
    
    trainset = BlackBoxDataset(which_set = 'train')
    testset = BlackBoxDataset(which_set = 'public_test')

    return trainset, testset
    
def get_layer_MLP():
    
    extraset = BlackBoxDataset( which_set = 'extra')
    
    processor = Standardize();
    
    processor.apply(extraset,can_fit=True)
    
    trainset = BlackBoxDataset( which_set = 'train',
                                start = 0,
                                stop = 900,
                                preprocessor = processor,
                                fit_preprocessor = True,
                                fit_test_preprocessor = True,
                                )
    
    validset = BlackBoxDataset( which_set = 'train',
                                start = 900,
                                stop = 1000 ,
                                preprocessor = processor,
                                fit_preprocessor = True,
                                fit_test_preprocessor = False,
                                )
    
    dropCfg = { 'input_include_probs': { 'h0' : .8 } ,
                'input_scales': { 'h0': 1.}
              }
    
    config = { 'learning_rate': .05,
                'init_momentum': .00,
                'cost' : Dropout(**dropCfg), 
                'monitoring_dataset':  { 'train' : trainset,
                                         'valid' : validset
                                        },
                'termination_criterion': MonitorBased(channel_name='valid_y_misclass',N=100,prop_decrease=0),
                'update_callbacks': None
              }
     
    config0 = {
                'layer_name': 'h0',
                'num_units': 1875,
                'num_pieces': 2,
                'irange': .05,
                # Rather than using weight decay, we constrain the norms of the weight vectors
                'max_col_norm': 2.
    }
    
    config1 = {
                'layer_name': 'h1',
                'num_units': 700,
                'num_pieces': 2,
                'irange': .05,
                # Rather than using weight decay, we constrain the norms of the weight vectors
                'max_col_norm': 2.
    }
    
    sftmaxCfg = {
                'layer_name': 'y',
                'init_bias_target_marginals': trainset,
                # Initialize the weights to all 0s
                'irange': .0,
                'n_classes': 9
            }
    
    l1 = Maxout(**config0)
    l2 = Maxout(**config1)
    l3 = Softmax(**sftmaxCfg)

    train_algo = SGD(**config)
    model = MLP(batch_size=75,layers=[l1,l2,l3],nvis=1875)
    return Train(model = model,
            dataset = trainset,
            algorithm = train_algo,
            extensions = None, 
            save_path = "maxout_best_model.pkl",
            save_freq = 1)

def main():

    model = get_layer_MLP()

    #supervised training
    model.main_loop()


if __name__ == '__main__':
    main()