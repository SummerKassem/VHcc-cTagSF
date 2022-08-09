import sys
import os
import uproot as uproot
import numpy as np
import awkward as ak
import pandas as pd

#import matplotlib.pyplot as plt
#import coffea.hist as hist

import gc

import torch
import torch.nn as nn

from focal_loss import FocalLoss

import warnings
warnings.filterwarnings("ignore", message="torch.distributed.reduce_op is deprecated, please use torch.distributed.ReduceOp instead")

minima = np.load('/nfs/dust/cms/user/summer/additional_files/minima.npy')
default = 0.001

defaults_per_variable = minima - default
threshold = 0.001

no_features = 67    
included_features = np.arange(0,no_features)

def cleandataset(f, isMC):
    
    feature_names = [k for k in f['Events'].keys() if  (('Jet_eta' == k) or ('Jet_pt' == k) or ('Jet_DeepCSV' in k))]

    # tagger output to compare with later and variables used to get the truth output
    feature_names.extend(('Jet_btagDeepB_b','Jet_btagDeepB_bb', 'Jet_btagDeepC','Jet_btagDeepL'))
    if isMC == True:
        feature_names.extend(('Jet_nBHadrons', 'Jet_hadronFlavour'))
    #print(feature_names)
    #print(len(feature_names)) => 71
    
    # go through a specified number of events, and get the information (awkward-arrays) for the keys specified above, place in data
    for data in f['Events'].iterate(feature_names, step_size=f['Events'].num_entries, library='ak'):
        break
    
    # creating an array to store all the columns with their entries per jet, flatten per-event -> per-jet
    datacolumns = np.zeros((len(feature_names)+1, len(ak.flatten(data['Jet_pt'], axis=1))))
    #print(f"length of datacolumns:{len(datacolumns)}")
    
    for featureindex in range(len(feature_names)):
        a = ak.flatten(data[feature_names[featureindex]], axis=1) # flatten along first inside to get jets
        datacolumns[featureindex] = ak.to_numpy(a)
    #print(f"shape of datacolumns:{np.shape(datacolumns)}")
    
    if isMC == True:
        nbhad = ak.to_numpy(ak.flatten(data['Jet_nBHadrons'], axis=1))
        hadflav = ak.to_numpy(ak.flatten(data['Jet_hadronFlavour'], axis=1))

        target_class = np.full_like(hadflav, 3)                                                      # udsg
        target_class = np.where(hadflav == 4, 2, target_class)                                       # c
        target_class = np.where(np.bitwise_and(hadflav == 5, nbhad > 1), 1, target_class)            # bb
        target_class = np.where(np.bitwise_and(hadflav == 5, nbhad <= 1), 0, target_class)           # b, lepb

        #print(np.unique(target_class))

        datacolumns[len(feature_names)] = target_class
        #print(np.unique(datacolumns[len(feature_names)]))
        
    datavectors = datacolumns.transpose()
    #print(np.unique(datavectors[:,len(feature_names)])) # if it is MC will contain jet-flavour otherwise 0
    #print(np.shape(datavectors))

    for j in range(no_features):
        datavectors[:, j][datavectors[:, j] == np.nan]  = defaults_per_variable[j]
        datavectors[:, j][datavectors[:, j] <= -np.inf] = defaults_per_variable[j]
        datavectors[:, j][datavectors[:, j] >= np.inf]  = defaults_per_variable[j]
        datavectors[:, j][datavectors[:, j] == -999]    = defaults_per_variable[j] 

    alldata = datavectors

    # featre[64] => jetNSelectedTRacks    
    # alldata[:,64] <= 0 means that in this jet there is less than 1 track!
    # so if there is at least one jet, leave it alone, otherwise give it default
    for track0_vars in [6,12,22,29,35,42,50]:
        alldata[:,track0_vars][alldata[:,64] <= 0] = defaults_per_variable[track0_vars]
    for track0_1_vars in [7,13,23,30,36,43,51]:
        alldata[:,track0_1_vars][alldata[:,64] <= 1] = defaults_per_variable[track0_1_vars]
    for track01_2_vars in [8,14,24,31,37,44,52]:
        alldata[:,track01_2_vars][alldata[:,64] <= 2] = defaults_per_variable[track01_2_vars]
    for track012_3_vars in [9,15,25,32,38,45,53]:
        alldata[:,track012_3_vars][alldata[:,64] <= 3] = defaults_per_variable[track012_3_vars]
    for track0123_4_vars in [10,16,26,33,39,46,54]:
        alldata[:,track0123_4_vars][alldata[:,64] <= 4] = defaults_per_variable[track0123_4_vars]
    for track01234_5_vars in [11,17,27,34,40,47,55]:
        alldata[:,track01234_5_vars][alldata[:,64] <= 5] = defaults_per_variable[track01234_5_vars]

    # feature[65] => Jet_DeepCSV_jetNTracksEtaRel :: the number of tracks for which eta_rel (between track and jet axis) is available
    alldata[:,18][alldata[:,65] <= 0] = defaults_per_variable[18]
    alldata[:,19][alldata[:,65] <= 1] = defaults_per_variable[19]
    alldata[:,20][alldata[:,65] <= 2] = defaults_per_variable[20]
    alldata[:,21][alldata[:,65] <= 3] = defaults_per_variable[21]

    # features[41,48,49,56] => features which are given for the first track above the charm mass
    for AboveCharm_vars in [41,48,49,56]:
        alldata[:,AboveCharm_vars][alldata[:,AboveCharm_vars]==-1] = defaults_per_variable[AboveCharm_vars] 
    
    datacls = [i for i in range(0,no_features)]

    if isMC == True:
        datacls.append(73)

    dataset = alldata[:, datacls]
    #print(f"Shape of dataset:{np.shape(dataset)}") # no_jets x no_features
    
    return dataset

def preprocess(rootfile, isMC):
    print('Doing starting clean/prep, isMC: ',isMC)
    
    dataset_input_target = cleandataset(uproot.open(rootfile), isMC)
    #print(f"Shape of dataset returned from cleaning:{np.shape(dataset_input_target)}")

    # use the global variables do not declare local new ones
    global no_features
    global included_features
    
    # if model should be ablated load correct ranking based on weighingMethod 
    if "ablated" in weighingMethod:
        
        ablationSpecs   = weighingMethod.split('_')
        topBottom       = ablationSpecs[1]
        no_excluded_features    = ablationSpecs[2]
        rawDistorted    = 'raw' if 'raw' in ablationSpecs[3] else 'distorted'
        ablatedModel    = 'Nominal' if 'Nominal' in ablationSpecs[3] else 'FGSM'
        epsilon         = '0.0' if ablationSpecs[3] == 'rawNominal' else '0.01'

        print(f"\nModel should be {ablationSpecs[0]}, by removing the {topBottom} {no_excluded_features} ranking features as evaluated using {rawDistorted} inputs on {ablatedModel} model\n")
        
        # load ranking into a dataframe df, the df will have an index and two columns: feature_name & ranking
        # df.index is the original index of the feature that matches dataset_input_target 
        df = pd.read_pickle(f'/nfs/dust/cms/user/summer/additional_files/feature_ranking/IG_DefaultBase/{rawDistorted}{ablatedModel}/combined-ranking_1000000-Jets_mean_TT_File_{rawDistorted}{ablatedModel}_Mode_{epsilon}_PARAM.pkl')
        # sort values descendingly/ascendingly depending on whether ablation will be done by removing the top/bottom features
        df = df.sort_values('ranking', ascending = False if topBottom=='top' else True)
        # now df.index contains the sorted indicies of the feature based on their ranking
        excluded_features = np.array(df.index[0:int(no_excluded_features)].values)
        included_features = np.array(df.index[int(no_excluded_features):].values)

        no_features = len(included_features) if len(included_features) < no_features else no_features

    inputs          = torch.Tensor(dataset_input_target[:,included_features])
    scaled_defaults = np.zeros_like(defaults_per_variable[included_features])
    scalers         = [torch.load(f'/nfs/dust/cms/user/summer/additional_files/scalers/scaler_{i}_with_default_{default}.pt') for i in included_features]

    # targets only make sense for MC, but nothing 'breaks' when calling it on Data (the last column is different though)
    targets = torch.Tensor(dataset_input_target[:,-1]).long()

    if (isMC & isInteractive):
        print(f"Possible Targets are: {torch.unique(targets)}") 

    for i in range(len(included_features)): 
        # use already calculated scalers (same for all files),
        # for the calculation, only train samples that are non-defaults were used
        #scaler = StandardScaler().fit(inputs[:,i][inputs[:,i]!=defaults_per_variable[i]].reshape(-1,1))
        
        inputs[:,i]         = torch.Tensor(scalers[i].transform(inputs[:,i].reshape(-1,1)).reshape(1,-1))
        scaled_defaults[i]  = scalers[i].transform(defaults_per_variable[i].reshape(-1,1)).reshape(1,-1)

    scaled_defaults = torch.Tensor(scaled_defaults)

    return inputs, targets, scaled_defaults

def apply_noise(sample, magn=1e-2,offset=[0], scaled_defaults_per_variable=[]):
    with torch.no_grad():
        device = torch.device("cpu")
        
        size_of_sample = len(sample)
        noise = torch.Tensor(np.random.normal(offset,magn,(size_of_sample,67))).to(device)
        xadv = sample + noise
        
        integervars = [59,63,64,65,66]

        for variable in integervars:
            xadv[:,variable] = sample[:,variable]

        for i in range(67):
            defaults = abs(sample[:,i].cpu() - scaled_defaults_per_variable[i].cpu()) <= threshold   # creates an array of true/false which if sample == scaled_default then True otherwise False
            
            if torch.sum(defaults) != 0: 
                xadv[:,i][defaults] = sample[:,i][defaults]

        return xadv

def fgsm_attack(epsilon=1e-2,sample=None,targets=None,reduced=True, scaled_defaults_per_variable=[]):
    device = torch.device("cpu")
    xadv = sample.clone().detach()
    
    # calculate the gradient of the model w.r.t. the *input* tensor:
    # first we tell torch that x should be included in grad computations
    xadv.requires_grad = True
    
    # from the undisturbed predictions, both the model and the criterion are already available and can be used here again; it's just that they were each part of a function, so not
    # automatically in the global scope
    
    global model
    global criterion
    
    # then we just do the forward and backwards pass as usual:
    preds = model(xadv)
    #print(targets)
    #print(torch.unique(targets))
    #print(preds)
    loss = criterion(preds, targets).mean()
    # maybe add sample weights here as well for the ptetaflavloss weighting method
    model.zero_grad()
    loss.backward()
    
    with torch.no_grad():
        #now we obtain the gradient of the input. It has the same dimensions as the tensor xadv, and it "points" in the direction of increasing loss values.
        dx = torch.sign(xadv.grad.detach())
        
        #so, we take a step in that direction!
        xadv += epsilon*torch.sign(dx)
        
        #remove the impact on selected variables. This is nessecary to avoid problems that occur otherwise in the input shapes.
        if reduced:
            integervars = [59,63,64,65,66]

            for variable in integervars:
                xadv[:,variable] = sample[:,variable]

            for i in range(67):
                #defaults = abs(scalers[i].inverse_transform(sample[:,i].cpu()) - defaults_per_variable[i]) < 0.001   # "floating point error" --> allow some error margin
                defaults = abs(sample[:,i].cpu() - scaled_defaults_per_variable[i].cpu()) <= threshold   # creates an array of true/false which if sample == scaled_default then True otherwise False

                if torch.sum(defaults) != 0:
                    xadv[:,i][defaults] = sample[:,i][defaults]

        return xadv.detach()

def predict(inputs, method):
    
    global model
    global criterion

    with torch.no_grad():
        device = torch.device("cpu")
        
        model = nn.Sequential(nn.Linear(no_features, 100),
                          nn.ReLU(),
                          nn.Dropout(0.1),
                          nn.Linear(100, 100),
                          nn.ReLU(),
                          nn.Dropout(0.1),
                          nn.Linear(100, 100),
                          nn.ReLU(),
                          nn.Dropout(0.1),
                          nn.Linear(100, 100),
                          nn.ReLU(),
                          nn.Dropout(0.1),
                          nn.Linear(100, 100),
                          nn.ReLU(),
                          nn.Linear(100, 4),
                          nn.Softmax(dim=1))
        
        if method == '_new':
            #allweights = compute_class_weight(
            #       'balanced',
            #        classes=np.array([0,1,2,3]), 
            #        y=targets.numpy().astype(int))
            #class_weights = torch.FloatTensor(allweights).to(device)
            #del allweights
            #gc.collect()
            #these classweights have been derived once for TTtoSemileptonic (the ones used for training)
            #class_weights = torch.FloatTensor(np.array([ 0.37333512, 24.65012434,  2.25474568,  1.1942229 ])).to(device)
            #criterion = nn.CrossEntropyLoss(weight=class_weights)
            criterion = nn.CrossEntropyLoss()

            modelPath = f'/nfs/dust/cms/user/summer/trained_models/saved_models/model_all_TT_350_epochs_v10_GPU_weighted_new_49_datasets_with_default_0.001.pt'
            #modelPath = f'/nfs/dust/cms/user/summer/trained_models/saved_models/normal_tr_278_-1/model_200_epochs_normal_tr_278_-1.pt'


        # ==========================================================================
        #
        #                               NEW: may_21
        #

        elif method == '_ptetaflavloss20':
            criterion = nn.CrossEntropyLoss(reduction='none')
            modelPath = f'/nfs/dust/cms/user/anstein/pretrained_models/model_124_epochs_v10_GPU_weighted_ptetaflavloss_20_datasets_with_default_0.001_-1.pt'

        elif method == '_ptetaflavloss278':
            criterion = nn.CrossEntropyLoss(reduction='none')
            modelPath = f'/nfs/dust/cms/user/anstein/pretrained_models/model_1_epochs_v10_GPU_weighted_ptetaflavloss_278_datasets_with_default_0.001_-1.pt'

        #
        #
        #
        # --------------------------------------------------------------------------


        # ==========================================================================
        #
        #                               NEW: as of June, 16th
        #

        elif method == '_ptetaflavloss_focalloss':
            # for focal loss: parameters
            alpha = None  # weights are handled differently, not with the focal loss but with sample weights if wanted
            gamma = 2
            criterion = FocalLoss(alpha, gamma, reduction='none')
            modelPath = f'/nfs/dust/cms/user/anstein/pretrained_models/model_200_epochs_v10_GPU_weighted_ptetaflavloss_focalloss_278_datasets_with_default_0.001_-1.pt'

        elif method == '_flatptetaflavloss_focalloss':
            # for focal loss: parameters
            alpha = None  # weights are handled differently, not with the focal loss but with sample weights if wanted
            gamma = 2
            criterion = FocalLoss(alpha, gamma, reduction='none')
            modelPath = f'/nfs/dust/cms/user/anstein/pretrained_models/model_200_epochs_v10_GPU_weighted_flatptetaflavloss_focalloss_278_datasets_with_default_0.001_-1.pt'

        #
        #
        #
        # --------------------------------------------------------------------------



        # ==========================================================================
        #
        #                               NEW: as of June, 25th
        #

        elif method == '_notflat_250_gamma2.0_alphaNone':
            # for focal loss: parameters
            alpha = None
            gamma = 2.0
            criterion = FocalLoss(alpha, gamma, reduction='none')
            modelPath = f'/nfs/dust/cms/user/anstein/pretrained_models/model_250_epochs_v10_GPU_weighted_ptetaflavloss_focalloss_278_datasets_with_default_0.001_-1.pt'

        elif method == '_flat_230_gamma2.0_alphaNone':
            # for focal loss: parameters
            alpha = None
            gamma = 2.0
            criterion = FocalLoss(alpha, gamma, reduction='none')
            modelPath = f'/nfs/dust/cms/user/anstein/pretrained_models/model_230_epochs_v10_GPU_weighted_flatptetaflavloss_focalloss_278_datasets_with_default_0.001_-1.pt'

        elif method == '_notflat_100_gamma20.0_alphaNone':
            # for focal loss: parameters
            alpha = None
            gamma = 20.0
            criterion = FocalLoss(alpha, gamma, reduction='none')
            modelPath = f'/nfs/dust/cms/user/anstein/pretrained_models/model_100_epochs_v10_GPU_weighted_ptetaflavloss_focalloss_gamma20.0_278_datasets_with_default_0.001_-1.pt'

        elif method == '_notflat_100_gamma4.0_alpha0.4,0.4,0.2,0.2':
            # for focal loss: parameters
            alpha = torch.Tensor([0.4,0.4,0.2,0.2])
            gamma = 4.0
            criterion = FocalLoss(alpha, gamma, reduction='none')
            modelPath = f'/nfs/dust/cms/user/anstein/pretrained_models/model_100_epochs_v10_GPU_weighted_ptetaflavloss_focalloss_gamma4.0_alpha0.4,0.4,0.2,0.2_278_datasets_with_default_0.001_-1.pt'

        elif method == '_flat_200_gamma25.0_alphaNone':
            # for focal loss: parameters
            alpha = None
            gamma = 25.0
            criterion = FocalLoss(alpha, gamma, reduction='none')
            modelPath = f'/nfs/dust/cms/user/anstein/pretrained_models/model_200_epochs_v10_GPU_weighted_flatptetaflavloss_focalloss_gamma25.0_278_datasets_with_default_0.001_-1.pt'

        #
        #
        #
        # --------------------------------------------------------------------------

        # ==========================================================================
        #
        #                               NEW: as of July, 8th
        #
        
        # only epoch 200
        elif method == '_notflat_200_gamma25.0_alphaNone_adv_tr_eps0.01':
            # for focal loss: parameters
            alpha = None
            gamma = 25.0
            criterion = FocalLoss(alpha, gamma, reduction='none')
            modelPath = f'/nfs/dust/cms/user/anstein/pretrained_models/adv_tr/model_200_epochs_v10_GPU_weighted_ptetaflavloss_focalloss_gamma25.0_adv_tr_eps0.01_278_datasets_with_default_0.001_-1.pt'
            
        # only epoch 200
        elif method == '_notflat_200_gamma25.0_alphaNone':
            # for focal loss: parameters
            alpha = None
            gamma = 25.0
            criterion = FocalLoss(alpha, gamma, reduction='none')
            modelPath = f'/nfs/dust/cms/user/anstein/pretrained_models/basic_tr/model_200_epochs_v10_GPU_weighted_ptetaflavloss_focalloss_gamma25.0_278_datasets_with_default_0.001_-1.pt'
            
        
        # special cases that can handle different epochs (checkpoints)    
        elif method.startswith('adv'):
            epoch = method.split('adv_tr_eps0.01_')[-1]
            # for focal loss: parameters
            alpha = None
            gamma = 25.0
            criterion = FocalLoss(alpha, gamma, reduction='none')
            modelPath = f'/nfs/dust/cms/user/anstein/pretrained_models/adv_tr/model_{epoch}_epochs_v10_GPU_weighted_ptetaflavloss_focalloss_gamma25.0_adv_tr_eps0.01_278_datasets_with_default_0.001_-1.pt'
            
        elif method.startswith('basic'):
            print("from here#1")
            epoch = method.split('basic_')[-1]
            # for focal loss: parameters
            alpha = None
            gamma = 25.0
            criterion = FocalLoss(alpha, gamma, reduction='none')
            modelPath = f'/nfs/dust/cms/user/summer/trained_models/saved_models/normal_tr_278_-1/model_200_epochs_normal_tr_278_-1.pt'

        elif method.startwith('ablated'):
            print("from here#2")
            ablationSpecs   = method.split('_')
            topBottom       = ablationSpecs[1]
            no_excluded_features    = ablationSpecs[2]
            origModel    = 'normal' if 'Nominal' in ablationSpecs[3] else 'adversarial'
            epsilon         = '' if 'Nominal' in ablationSpecs[3] else 'eps0.01_'

            criterion = FocalLoss(alpha = None, gamma = 25.0, reduction='none')

            modelPath = f'/nfs/dust/cms/user/summer/trained_models/saved_models/ablation_IG_DefaultBase_{topBottom}_{no_excluded_features}_{ablationSpecs[3]}_TT_{origModel}_tr_{epsilon}278_-1/'
            
        #
        #
        #
        # --------------------------------------------------------------------------

        # old
        else:
            criterion = nn.CrossEntropyLoss()
            #modelPath = f'/nfs/dust/cms/user/anstein/pretrained_models/model_all_TT_350_epochs_v10_GPU_weighted_as_is_49_datasets_with_default_0.001.pt'
            modelPath = f'/nfs/dust/cms/user/summer/trained_models/saved_models/normal_tr_278_-1/model_200_epochs_normal_tr_278_-1.pt'
        
        checkpoint = torch.load(modelPath, map_location=torch.device(device))
        model.load_state_dict(checkpoint["model_state_dict"])

        # marker
        # test for basic, adversarial then ablated normal and adversarial
        print(f"model loaded successfully")
        sys.exit()

        model.to(device)

        #evaluate network on inputs
        model.eval()
        
        return model(inputs).detach().numpy()

def calcBvsL(matching_predictions):
    global n_jets
    
    custom_BvL = np.where(((matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,3]) != 0) & (matching_predictions[:,0] >= 0) & (matching_predictions[:,0] <= 1) & (matching_predictions[:,1] >= 0) & (matching_predictions[:,1] <= 1) & (matching_predictions[:,2] >= 0) & (matching_predictions[:,2] <= 1) & (matching_predictions[:,3] >= 0) & (matching_predictions[:,3] <= 1), (matching_predictions[:,0]+matching_predictions[:,1])/(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,3]), (-1.0)*np.ones(n_jets))
    
    custom_BvL[(custom_BvL < 0.000001) & (custom_BvL > -0.000001)] = 0.000001
    custom_BvL[(np.isnan(custom_BvL)) | (np.isinf(custom_BvL))] = -1.0
    custom_BvL[custom_BvL > 0.99999] = 0.99999
    
    return custom_BvL

def calcBvsC(matching_predictions):
    global n_jets
    
    custom_BvC = np.where(((matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]) != 0) & (matching_predictions[:,0] >= 0) & (matching_predictions[:,0] <= 1) & (matching_predictions[:,1] >= 0) & (matching_predictions[:,1] <= 1) & (matching_predictions[:,2] >= 0) & (matching_predictions[:,2] <= 1) & (matching_predictions[:,3] >= 0) & (matching_predictions[:,3] <= 1), (matching_predictions[:,0]+matching_predictions[:,1])/(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]), (-1.0)*np.ones(n_jets))
    
    custom_BvC[(custom_BvC < 0.000001) & (custom_BvC > -0.000001)] = 0.000001
    custom_BvC[(np.isnan(custom_BvC)) | (np.isinf(custom_BvC))] = -1.0
    custom_BvC[custom_BvC > 0.99999] = 0.99999
    
    return custom_BvC
    
def calcCvsB(matching_predictions):
    global n_jets
    
    custom_CvB = np.where(((matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]) != 0) & (matching_predictions[:,0] >= 0) & (matching_predictions[:,0] <= 1) & (matching_predictions[:,1] >= 0) & (matching_predictions[:,1] <= 1) & (matching_predictions[:,2] >= 0) & (matching_predictions[:,2] <= 1) & (matching_predictions[:,3] >= 0) & (matching_predictions[:,3] <= 1), (matching_predictions[:,2])/(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]), (-1.0)*np.ones(n_jets))
    
    custom_CvB[(custom_CvB < 0.000001) & (custom_CvB > -0.000001)]  = 0.000001
    custom_CvB[(np.isnan(custom_CvB)) | (np.isinf(custom_CvB))]     = -1.0
    custom_CvB[custom_CvB > 0.99999]                                = 0.99999
    
    return custom_CvB
    
def calcCvsL(matching_predictions):
    global n_jets
    
    custom_CvL = np.where(((matching_predictions[:,2]+matching_predictions[:,3]) != 0) & (matching_predictions[:,0] >= 0) & (matching_predictions[:,0] <= 1) & (matching_predictions[:,1] >= 0) & (matching_predictions[:,1] <= 1) & (matching_predictions[:,2] >= 0) & (matching_predictions[:,2] <= 1) & (matching_predictions[:,3] >= 0) & (matching_predictions[:,3] <= 1), (matching_predictions[:,2])/(matching_predictions[:,2]+matching_predictions[:,3]), (-1.0)*np.ones(n_jets))
    
    custom_CvL[(custom_CvL < 0.000001) & (custom_CvL > -0.000001)] = 0.000001
    custom_CvL[(np.isnan(custom_CvL)) | (np.isinf(custom_CvL))] = -1.0
    custom_CvL[custom_CvL > 0.99999] = 0.99999
    
    return custom_CvL
    
def calcBvsL_legacy(predictions):  # P(b)+P(bb)/(P(b)+P(bb)+P(udsg))
    bvsl = (predictions[:,0]+predictions[:,1])/(1-predictions[:,2])
    bvsl[bvsl < 0.000001] = 0.000001
    bvsl[bvsl > 0.99999] = 0.99999
    return bvsl    
def calcBvsC_legacy(predictions):  # P(b)+P(bb)/(P(b)+P(bb)+P(c))
    bvsc = (predictions[:,0]+predictions[:,1])/(1-predictions[:,3])
    bvsc[bvsc < 0.000001] = 0.000001
    bvsc[bvsc > 0.99999] = 0.99999
    return bvsc   
def calcCvsB_legacy(predictions):  # P(c)/(P(b)+P(bb)+P(c))
    cvsb =  (predictions[:,2])/(predictions[:,0]+predictions[:,1]+predictions[:,2])
    cvsb[cvsb < 0.000001] = 0.000001
    cvsb[cvsb > 0.99999] = 0.99999
    cvsb[np.isnan(cvsb)] = -1
    return cvsb
def calcCvsL_legacy(predictions):  # P(c)/(P(udsg)+P(c))
    cvsl = (predictions[:,2])/(predictions[:,3]+predictions[:,2])
    cvsl[cvsl < 0.000001] = 0.000001
    cvsl[cvsl > 0.99999] = 0.99999
    cvsl[np.isnan(cvsl)] = -1
    return cvsl

if __name__ == "__main__":
    
    fullName, weighingMethod, condoroutdir, isInteractive = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]=='yes'

    if isInteractive:
        print(f"\nScript is being run interactively")

    parentDir = ""
    era = 2016

    print("\nWill open file %s \n"%(fullName))

    parentDirList = ["/106X_v2_17/","/106X_v2_17rsb2/","/106X_v2_17rsb3/"]

    for iParent in parentDirList:
        if iParent in fullName: 
            parentDir = iParent

    if parentDir == "": fullName.split('/')[8]+"/"

    # this is needed because both 2017 and 2018 appear in the new file names
    if "2017" in fullName: 
        era = 2017
    if "2018" and not "2017" in fullName: 
        era = 2018 

    sampName=   fullName.split(parentDir)[1].split('/')[0]
    channel =   sampName

    if "PFNano" in fullName:
        sampNo  =   fullName.split(parentDir)[1].split('/')[1]
    else: # for PostProc files
        sampNo =   fullName.split(parentDir)[1].split('/')[1].split('_')[-1]

    dirNo   =   fullName.split(parentDir)[1].split('/')[3][-1]
    flNo    =   fullName.split(parentDir)[1].split('/')[-1].rstrip('.root').split('_')[-1]
    outNo   =   "%s_%s_%s"%(sampNo,dirNo,flNo)

    if "_" in channel: 
        channel=channel.split("_")[0]

    if not 'Single' in channel and not 'Double' in channel and not 'EGamma' in channel:
        isMC = True
    else:
        isMC = False

    print("Using channel =",channel, "; isMC:", isMC, "; era: %d"%era, "; outNo:", outNo)

    global n_jets

    if isInteractive: 
        inputs, targets, scaled_defaults = preprocess(fullName, isMC)                  
    else:
        inputs, targets, scaled_defaults = preprocess('infile.root', isMC)

    n_jets = len(targets)

    # to check multiple epochs of a given weighting method at once (using always 3 epochs should make sense, as previous tests were done on raw/noise/FGSM = 3 different sets)
    if weighingMethod.startswith('_multi_'):
        letters = ['A','B','C']  # using the same three letters all the time means that the Analyzer code does not need to be updated for every possible epoch
        if 'basic' in weighingMethod:
            # basic training on raw inputs only
            wmethods = ['basic_'+e for e in (weighingMethod.split('_basic_')[-1]).split(',')]
        else:
            # adversarial training
            wmethods = ['adv_tr_eps0.01_'+e for e in (weighingMethod.split('_adv_tr_eps0.01_')[-1]).split(',')]

        print('Will run with these weighting methods & epochs:', wmethods)
        
        for i,wm in enumerate(wmethods):
            outputPredsdir = f"{letters[i]}_outPreds_%s.npy"%(outNo)
            outputCvsBdir  = f"{letters[i]}_outCvsB_%s.npy"%(outNo)
            outputCvsLdir  = f"{letters[i]}_outCvsL_%s.npy"%(outNo)
            outputBvsCdir  = f"{letters[i]}_outBvsC_%s.npy"%(outNo)
            outputBvsLdir  = f"{letters[i]}_outBvsL_%s.npy"%(outNo)
            
            predictions = predict(inputs, wm)
            
            bvl = calcBvsL(predictions)
            print('Raw bvl, bvc, cvb, cvl')
            print(min(bvl), max(bvl))
            np.save(outputBvsLdir, bvl)
            del bvl
            gc.collect()

            bvc = calcBvsC(predictions)
            print(min(bvc), max(bvc))
            np.save(outputBvsCdir, bvc)
            del bvc
            gc.collect()

            cvb = calcCvsB(predictions)
            
            print(min(cvb), max(cvb))
            np.save(outputCvsBdir, cvb)
            del cvb
            gc.collect()
            cvl = calcCvsL(predictions)
            
            print(min(cvl), max(cvl))
            np.save(outputCvsLdir, cvl)
            del cvl
            gc.collect()

            predictions[:,0][predictions[:,0] > 0.99999] = 0.99999
            predictions[:,1][predictions[:,1] > 0.99999] = 0.99999
            predictions[:,2][predictions[:,2] > 0.99999] = 0.99999
            predictions[:,3][predictions[:,3] > 0.99999] = 0.99999
            predictions[:,0][predictions[:,0] < 0.000001] = 0.000001
            predictions[:,1][predictions[:,1] < 0.000001] = 0.000001
            predictions[:,2][predictions[:,2] < 0.000001] = 0.000001
            predictions[:,3][predictions[:,3] < 0.000001] = 0.000001
            print('Raw b, bb, c, l min and max (after cutting over-/underflow)')
            print(min(predictions[:,0]), max(predictions[:,0]))
            print(min(predictions[:,1]), max(predictions[:,1]))
            print(min(predictions[:,2]), max(predictions[:,2]))
            print(min(predictions[:,3]), max(predictions[:,3]))
            np.save(outputPredsdir, predictions)
            del predictions
            gc.collect()
            
    # just one weighting method at a given epoch, but with Noise or FGSM attack applied to MC
    else:

        wm = weighingMethod
        
        outputPredsdir = "outPreds_%s.npy"%(outNo)
        outputCvsBdir = "outCvsB_%s.npy"%(outNo)
        outputCvsLdir = "outCvsL_%s.npy"%(outNo)
        outputBvsCdir = "outBvsC_%s.npy"%(outNo)
        outputBvsLdir = "outBvsL_%s.npy"%(outNo)

        noise_outputPredsdir = "noise_outPreds_%s.npy"%(outNo)
        noise_outputCvsBdir = "noise_outCvsB_%s.npy"%(outNo)
        noise_outputCvsLdir = "noise_outCvsL_%s.npy"%(outNo)
        noise_outputBvsCdir = "noise_outBvsC_%s.npy"%(outNo)
        noise_outputBvsLdir = "noise_outBvsL_%s.npy"%(outNo)

        fgsm_outputPredsdir = "fgsm_outPreds_%s.npy"%(outNo)
        fgsm_outputCvsBdir = "fgsm_outCvsB_%s.npy"%(outNo)
        fgsm_outputCvsLdir = "fgsm_outCvsL_%s.npy"%(outNo)
        fgsm_outputBvsCdir = "fgsm_outBvsC_%s.npy"%(outNo)
        fgsm_outputBvsLdir = "fgsm_outBvsL_%s.npy"%(outNo)

        #print("Saving into %s/%s"%(condoroutdir,sampName))
        
        predictions = predict(inputs, wm)
        print(f"shape of predictions:{np.shape(predictions)}")
        
        bvl = calcBvsL(predictions)
        
        print('Raw bvl, bvc, cvb, cvl')
        print(min(bvl), max(bvl))
        np.save(outputBvsLdir, bvl)
        del bvl
        gc.collect()
        
        bvc = calcBvsC(predictions)
        print(min(bvc), max(bvc))
        np.save(outputBvsCdir, bvc)
        del bvc
        gc.collect()

        cvb = calcCvsB(predictions)
        print(min(cvb), max(cvb))
        np.save(outputCvsBdir, cvb)
        del cvb
        gc.collect()

        cvl = calcCvsL(predictions)
        print(min(cvl), max(cvl))
        np.save(outputCvsLdir, cvl)
        del cvl
        gc.collect()

        for i in range(4):
            predictions[:,i][predictions[:,i] > 0.99999] = 0.99999  
            predictions[:,i][predictions[:,i] < 0.000001] = 0.000001  
        
        print('Raw b, bb, c, l min and max (after cutting over-/underflow)')
        for i in range(4):
            print(min(predictions[:,i]), max(predictions[:,i]))

        np.save(outputPredsdir, predictions)
        del predictions
        gc.collect()
        
        if isMC == True:
            noise_sample = apply_noise(inputs, magn=1e-2, offset=[0], scaled_defaults_per_variable=scaled_defaults)
            noise_preds = predict(noise_sample, wm)
            
            print('Noise bvl, bvc, cvb, cvl')
            noise_bvl = calcBvsL(noise_preds)
            print(min(noise_bvl), max(noise_bvl))
            np.save(noise_outputBvsLdir, noise_bvl)
            del noise_bvl
            gc.collect()

            noise_bvc = calcBvsC(noise_preds)
            print(min(noise_bvc), max(noise_bvc))
            np.save(noise_outputBvsCdir, noise_bvc)
            del noise_bvc
            gc.collect()

            noise_cvb = calcCvsB(noise_preds)
            print(min(noise_cvb), max(noise_cvb))
            np.save(noise_outputCvsBdir, noise_cvb)
            del noise_cvb
            gc.collect()

            noise_cvl = calcCvsL(noise_preds)
            print(min(noise_cvl), max(noise_cvl))
            np.save(noise_outputCvsLdir, noise_cvl)
            del noise_cvl

            for i in range(4):
                noise_preds[:,i][noise_preds[:,i] > 0.99999] = 0.99999  
                noise_preds[:,i][noise_preds[:,i] < 0.000001] = 0.000001  
        

            print('Noise b, bb, c, l min and max (after cutting over-/underflow)')
            for i in range(4):
                print(min(noise_preds[:,i]), max(noise_preds[:,i]))
            
            np.save(noise_outputPredsdir, noise_preds)
            del noise_preds
            gc.collect()

            fgsm_sample = fgsm_attack(epsilon=1e-2,sample=inputs,targets=targets,reduced=True, scaled_defaults_per_variable = scaled_defaults)
            fgsm_preds = predict(fgsm_sample, wm)
            
            fgsm_bvl = calcBvsL(fgsm_preds)
            print('FGSM bvl, bvc, cvb, cvl')
            print(min(fgsm_bvl), max(fgsm_bvl))
            np.save(fgsm_outputBvsLdir, fgsm_bvl)
            del fgsm_bvl
            gc.collect()

            fgsm_bvc = calcBvsC(fgsm_preds)
            print(min(fgsm_bvc), max(fgsm_bvc))
            np.save(fgsm_outputBvsCdir, fgsm_bvc)
            del fgsm_bvc
            gc.collect()

            fgsm_cvb = calcCvsB(fgsm_preds)
            print(min(fgsm_cvb), max(fgsm_cvb))
            np.save(fgsm_outputCvsBdir, fgsm_cvb)
            del fgsm_cvb
            gc.collect()

            fgsm_cvl = calcCvsL(fgsm_preds)
            print(min(fgsm_cvl), max(fgsm_cvl))
            np.save(fgsm_outputCvsLdir, fgsm_cvl)
            del fgsm_cvl

            for i in range(4):
                fgsm_preds[:,i][fgsm_preds[:,i] > 0.99999] = 0.99999  
                fgsm_preds[:,i][fgsm_preds[:,i] < 0.000001] = 0.000001
                
            print('FGSM b, bb, c, l min and max (after cutting over-/underflow)')
            for i in range(4):
                print(min(fgsm_preds[:,i]), max(fgsm_preds[:,i]))
                
            np.save(fgsm_outputPredsdir, fgsm_preds)
            del fgsm_preds
            gc.collect()

'''
percentage_of_default_samples = torch.zeros_like(scaled_defaults_per_variable)
        
index = 1
what_to_plot = sample[:,index].numpy()

histogram = hist.Hist("Jets",
                        hist.Cat("sample","sample name"),
                        hist.Bin("prop",f"variable number {index}", 100, min(what_to_plot)-0.2, max(what_to_plot)+0.2))

print(f"Index is:{index}, scaled_default is:{scaled_defaults_per_variable[index]}")

defaults1 = abs(sample[:,index].cpu() - scaled_defaults_per_variable[index].cpu()) < 0.001   
defaults2 = abs(sample[:,index].cpu() - scaled_defaults_per_variable[index].cpu()) < 0.0001  
defaults3 = abs(sample[:,index].cpu() - scaled_defaults_per_variable[index].cpu()) == 0 

print(f"Number of samples:")
print(f"abs(difference) < 0.001 :: {len(sample[:,index][defaults1])}")
print(f"abs(difference) < 0.0001 :: {len(sample[:,index][defaults2])}")
print(f"abs(difference) == 0 :: {len(sample[:,index][defaults3])}")

histogram.fill(sample = "all",      prop = what_to_plot)
histogram.fill(sample = "< 0.001",  prop = sample[:,index][defaults1].numpy())
histogram.fill(sample = "< 0.0001", prop = sample[:,index][defaults2].numpy())
histogram.fill(sample = "== 0",     prop = sample[:,index][defaults3].numpy())

_, ax1 = plt.subplots(1,1,figsize=[10,6])
hist.plot1d(histogram, overlay='sample', ax = ax1)
ax1.set_yscale('log')
ax1.autoscale()
plt.show()
                    
sys.exit()

index = 21
threshold = 0

predictions = predict(inputs, weighingMethod)
raw_sample      = inputs[:,index].numpy()
noise_sample    = apply_noise(inputs, magn=1e-2, offset=[0], scaled_defaults_per_variable=scaled_defaults)[:,index].numpy()
fgsm_sample     = fgsm_attack(epsilon=1e-2, sample=inputs, targets=targets, reduced=True, scaled_defaults_per_variable = scaled_defaults)[:,index].numpy()

minimum = min(min(min(raw_sample),min(noise_sample)), min(fgsm_sample)) - 0.1
maximum = max(max(max(raw_sample),max(noise_sample)), max(fgsm_sample)) + 0.1

histogram = hist.Hist("Jets",
                        hist.Cat("sample","sample name"),
                        hist.Bin("prop",f"variable number {index}", 100, minimum, maximum))

print(f"Variable is:{index}, scaled_default is:{scaled_defaults[index]}")

histogram.fill(sample = "raw",   prop = raw_sample)
histogram.fill(sample = "fgsm",  prop = fgsm_sample)
histogram.fill(sample = "noise", prop = noise_sample)

_, ax1 = plt.subplots(1,1,figsize=[10,6])
hist.plot1d(histogram, overlay='sample', ax = ax1)
ax1.set_yscale('log')
ax1.autoscale()
plt.title(f"Threshold is {threshold}")
plt.show()
                    
sys.exit()
'''