
def init_value_from_config(config, search_hp):
    """
    Get default value from config 
    """
    out_init_dict = dict()
    for key in search_hp.keys():
        key_list = key.split('#')
        start = config
        for item in key_list:
            if not isinstance(start[item], dict):
                out_init_dict[key] = start[item]
            else:
                start = start[item]
    return out_init_dict
