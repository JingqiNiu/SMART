from collections import OrderedDict
import torch


def load_pretrain_model(model, checkpoint_path, key='state_dict', save_dir=None):
    """
    load pretrain checkpoint file for model

    Args:
        model (nn.Module): 
        checkpoint_path (str):  The path of the checkpoint.
        key (str, optional): the key in checkpoint file. Defaults to 'state_dict'.
        save_dir (str, optional): the path to save checkpoint. Defaults to None.
    Returns:
        [nn.Module]: model
    """
    model_dict = model.state_dict()
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if key:
        if key in state_dict:# lightning default key is 'state_dict'
            state_dict = state_dict[key]
        else:
            key_list = list(filter(lambda x: 'state_dict' in x, state_dict.keys()))
            assert len(key_list) > 0
            print(f"key of checkpoint is : {key_list[0]} ")
            state_dict = state_dict[key_list[0]]

    checkpoint_dict = OrderedDict()
    first_layer_name = list(state_dict.keys())[0]

    if first_layer_name.startswith('module.'):
        start_index = 7
    elif first_layer_name.startswith('model.'):
        start_index = 6
    else:
        start_index = 0

    for k, v in state_dict.items():
        name = k[start_index:]  # remove 'prefix'
        checkpoint_dict[name] = v
    # load params
    if model_dict[list(model_dict.keys())[-1]].shape == checkpoint_dict[list(checkpoint_dict.keys())[-1]].shape:
        model.load_state_dict(checkpoint_dict)
    else:
        # delect fc.bias checkpoint_dict
        checkpoint_dict.pop(list(checkpoint_dict)[-1])
        # delect fc.weight checkpoint_dict
        checkpoint_dict.pop(list(checkpoint_dict)[-1])
        msg = model.load_state_dict(checkpoint_dict, strict=False)
        print('load msg', msg)
    print(f'load model from {checkpoint_path}')
    if save_dir:  # save checkpoint,key = 'model_state_dict'
        torch.save({'model_state_dict': model.state_dict()}, save_dir)
        print(f'save mode : {save_dir}')
    return model
