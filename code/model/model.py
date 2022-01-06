import model.enabl3s_feature_model as enabl3s_feature_model
import model.dsads_feature_model as dsads_feature_model
import model.UTD_model as UTD_model
import torch
from datasets.dataset_read import sensor_idx_to_feature_indices
from thop import profile

def Generator(dataset = 'DSADS', sensor_idx = 0):
    input_size = len(sensor_idx_to_feature_indices(dataset_name=dataset, sensor_idx=sensor_idx))
    if 'DSADS_feature' == dataset:
        model = dsads_feature_model.Generator(input_size=input_size)
    elif 'ENABL3S_feature' == dataset:
        model = enabl3s_feature_model.Generator(input_size=input_size)
    else:
        model = UTD_model.Generator()
    '''
    Calculate the model size
    '''
    # input = torch.randn(1, input_size)
    # flops, params = profile(model, inputs=(input,))
    # print('Generator GFLOPs: {}, parameters: {}'.format(flops/1e9, params),)

    return model

def Classifier(dataset = 'DSADS', sensor_idx = 0, n_c = 1):
    input_size = len(sensor_idx_to_feature_indices(dataset_name=dataset, sensor_idx=sensor_idx))
    if 'DSADS' == dataset:
        model = dsads_model.Classifier()
    elif 'ENABL3S' == dataset:
        model = enabl3s_model.Classifier()
    elif 'DSADS_feature' == dataset:
        model = dsads_feature_model.Classifier(input_size=input_size, n_c = n_c)
    elif 'UTD_feature' == dataset:
        model = UTD_feature_model.Classifier()
    elif 'UTD' == dataset:
        model = UTD_model.Classifier()
    else:
        model = enabl3s_feature_model.Classifier(input_size=input_size, n_c = n_c)
    '''
        Calculate the model size
    '''
    # input = torch.randn(1, 128)
    # flops, params = profile(model, inputs=(input,))
    # print('Classifier GFLOPs: {}, parameters: {}'.format(flops/1e9, params),)

    return model

def DomainPredictor(dataset = 'DSADS'):
    if 'DSADS' == dataset:
        return dsads_model.DomainPredictor()
    elif 'ENABL3S' == dataset:
        return enabl3s_model.DomainPredictor()
    elif 'DSADS_feature' == dataset:
        return dsads_feature_model.DomainPredictor()
    elif 'UTD_feature' == dataset:
        return UTD_feature_model.DomainPredictor()
    elif 'UTD' == dataset:
        return UTD_model.DomainPredictor()
    else:
        return enabl3s_feature_model.DomainPredictor()



