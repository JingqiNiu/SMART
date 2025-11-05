import yaml
import copy
abs_dir = '/'.join(__file__.split('/')[:-2]) + '/'
with open( abs_dir + 'data_lib/path_infor.yaml', 'r') as ff:
    dataset_all_infor = yaml.load(ff, Loader=yaml.FullLoader)



def data_str_list_to_dict(input_name):
    """
    Convert different data types to dict . key is dataset name , value is sampling rate. 

    Args:
        input_name (list ,str ,dict): different data types

    Returns:
        dict 
    """
    output = dict()
    if isinstance(input_name, str):
        output[input_name] = 1.0
    elif isinstance(input_name, list):
        for key in input_name:
            output[key] = 1.0
    else:
        assert isinstance(input_name, dict)
        output = input_name
    return output


def one_dataset_infor(data_name, scale, label_name, dataset_all_infor=dataset_all_infor , csv_path = None):
    """
    get one data information from yaml

    Args:
        data_name (str): dataset name
        scale (int ,float , list): The sampling ratio of the data. 1.0 is no sampling
        label_name (str): the normalized label name
        dataset_all_infor  (dict, optional): the data information from 'path_infor.yaml'.
    Returns:
        csv_infor (dict)
    """
    item = dataset_all_infor[data_name]
    if csv_path != None:
        item['csv_path'] = csv_path
        print('REMODIFY THE LOAD csv_path to', csv_path)
    csv_infor = copy.deepcopy(item)

    csv_infor.setdefault('img_path_col','path')
    csv_infor.setdefault('data_weight_col','data_weight')
    csv_infor['scale'] = scale if scale else 1.0

    if isinstance(label_name ,str ):
        label_name = [label_name]

    label_list = []
    for name in label_name:
        label_list.append( item.get(name,name)) # default dr:dr
    
    csv_infor['col_in_csv']  = label_list  
    csv_infor['label_name']  = label_name
    return csv_infor

def to_data_define( data_name ,dataset_define):
    """
     converts the character dataset name to a complex predefined composite dataset dictionary
    """
    assert isinstance(dataset_define ,dict )
    if isinstance(data_name, str) and (data_name in dataset_define):
        return dataset_define[data_name]
    else:
        return data_name

def get_data_infor(data_names, label_name , dataset_define = None, csv_path = None):
    """
    get all data information from yaml

    Args:
        data_names (str ,list,dict):  dataset names with scale information
        label_name (str): the normalized label name
        dataset_define(dict or None):contains a combination of predefined data sets
    Returns:
        (list) : csv information list
    """
    if dataset_define is not None:
        data_names = to_data_define(data_names ,dataset_define)

    data_names_dict = data_str_list_to_dict(data_names)

    csv_infor_list = []
    for key in data_names_dict:
        temp_csv_infor = one_dataset_infor(key, scale=data_names_dict[key], label_name=label_name, csv_path = csv_path)
        csv_infor_list.append(temp_csv_infor)

    return csv_infor_list
