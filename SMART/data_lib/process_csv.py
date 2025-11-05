import numpy as np
import pandas as pd
# more_col = ['gender_norm', 'age_norm', '糖尿病病史', '高血压病史','脑血管病病史','冠心病病史','脂代谢异常病史','中心分形维数', '中心动静脉当量比', '中心弯曲度',]
# more_col = ['gender_norm', 'age_norm', '糖尿病病史', '高血压病史','脑血管病病史','冠心病病史','脂代谢异常病史']
# more_col = ['gender_norm', 'age_norm']
# more_col = ['gender_norm', 'age_norm', '糖尿病病史','冠心病病史','脂代谢异常病史']
# more_col = ['gender_norm', 'age_norm', '糖尿病病史', '高血压病史','脑血管病病史','冠心病病史','脂代谢异常病史','超广角分形维数', '超广角动静脉当量比', '超广角弯曲度',]
# print('根据输入的超广角或者中心图像的不同，请切换合适的 more_col',more_col) # 可能是额外的数据输入的地方？
def csv_col_map(col_data:pd.Series,map_table:dict ):
    """
    Map the column according to "map_table"
    """
    for val in col_data.unique():
        if val not in map_table  and (~ np.isnan(val)):
            map_table[val] = val
    col_data = col_data.map(map_table)    # relabel 
    return col_data,map_table   

def basic_read_csv(cfg, data_cfg):
    """
    cfg(dict): 
    
    return: pd.DataFrame 
    """
    col_in_csv = cfg['col_in_csv']
    path_col = cfg['img_path_col']
    csv_path = cfg['csv_path']
    data_weight_col = cfg.get('data_weight_col','data_weight')
    try:
        csv = pd.read_csv( csv_path )
    except:
        csv = pd.read_csv( csv_path ,encoding= 'gbk' )
    if data_weight_col in csv.columns:
        csv['data_weight'] = csv[data_weight_col].copy()
    else:
        csv['data_weight'] = csv[col_in_csv[0]] #! only use the first label for data weight
   
    rename_map = { path_col :'path'}
    for name ,col_csv in zip(cfg['label_name'],col_in_csv):
        rename_map[col_csv] = 'label'  # rename  dr --> label

        if col_csv.startswith('fake'):
            csv[col_csv] = 0

        map_key = f'{name}_map'
        if map_key in cfg:
            map_table = cfg[map_key]
            csv[col_csv],map_table =csv_col_map(csv[col_csv],map_table ) # adjustment of classification categories
            print(f'{csv_path} use label map \n {map_key} is #{map_table}#')
    more_col = data_cfg['more_col']
    sel_col = [path_col,'data_weight'] + col_in_csv   + more_col
    print('$$$$$$$$$',csv.columns)
    csv = csv[sel_col].rename( rename_map, axis =1)

    csv['path'] = cfg['img_dir'] +  csv[ 'path' ]  # Broadcasting
     
    data = csv[['path','label','data_weight',] + more_col] #!
    data = data.fillna(0) # if we set {-1: None} in label_map ,we can drop -1
    return data


def get_sample_weight(label, alpha=1.0):
    """
    get weight of every sample,
    alpha == 0, natural distribution
    alpha == 1, uniform distribution

    Args:
        label ([np.ndarray or list]): [description]
        alpha (float, optional): range [0, 1].
                When `alpha` == 0, the  class distribution will be natural distribution
                When `alpha` == 1, the  class distribution will be uniform distribution
                Defaults to 1.0.
    Returns:
        [np.ndarray]: weight of every sample
    """
    n_samples = len(label)
    if alpha  == 0:
        return np.ones(n_samples)

    if isinstance(label, list):
        label = np.array(label)
    elif (not isinstance(label,np.ndarray)):
        label = label.to_numpy()

    if label.ndim == 2:
        label = label [:,0] #! only use the first channels
    if isinstance(alpha,str) and alpha.startswith('reduce_gaussian'):
        print('reduce_gaussian')
        cut = np.percentile(label,[10,25,75,90])
        label = np.digitize(label,bins =cut)
        alpha = float(alpha.split('#')[-1])
    else:
        label = label.astype(int)
    count_pre_class = np.bincount(label)
    n_class = count_pre_class.shape[0]  # class_number

    original_weights = np.ones(n_class)
    uniform_weights = n_samples / count_pre_class / n_class
    final_weights = original_weights * (1-alpha) + uniform_weights * alpha
    weight_pre_sample = final_weights[label]
    return weight_pre_sample


def sample_df_fun(df, label_name, scale, seed=0):
    """
    According to scale and label_name ,sample dataframe
    Args:
        df (pd.dataframe):
        label_name (str): the normalized label name
        scale (int ,float,list): sampling quantity, sampling ratio, stratified sampling
        seed(int): seed for df.sample
    Returns:
        dataframe after sample
    """
    def sample_int_float(df, number):
        if number == 1.0:  # avoid be shuffled
            return df
        if isinstance(number, int):
            if number > df.shape[0]:
                print(f'sample number:{number} > actual number :{df.shape[0]}. #set replace = True')
                replace = True
            else:
                replace = False
            return df.sample(n=number, replace=replace, random_state=seed)

        if isinstance(number, float):
            replace = True if number > 1 else False
            return df.sample(frac=number, replace=replace, random_state=seed)

    def group_sample_fun(group, scale):
        name = group.name
        number = scale[name]
        return sample_int_float(group, number)

    if isinstance(scale, list):
        sample_df = df.groupby(by=label_name, group_keys=False).apply(
            group_sample_fun, scale)
    else:
        sample_df = sample_int_float(df, scale)
    return sample_df


def merge_multiple_dataset(csv_infor, sample_seed=0, data_cfg = None):
    """
    according csv_infor ,read ,sample, merge csv.
    """
    final_df = pd.DataFrame()
    for item in csv_infor:
        data  = basic_read_csv(item, data_cfg)  # read csv
        data = sample_df_fun(data, 'label', item['scale'], sample_seed)  # sample
        final_df = pd.concat([final_df, data], axis=0)  # merge

    final_labels = final_df['label'].to_numpy()
    if final_labels.ndim == 1 or  final_labels.shape[1] == 1:
        unique_val, count = np.unique(final_labels, return_counts=True)
        if len(unique_val) <50:
            print(f' : {unique_val}, count: {count}.')
    else:
        for channel_num in range(final_labels.shape[1]):
            unique_val, count = np.unique(final_labels[:,channel_num], return_counts=True)
            if len(unique_val) <50:
                print(f'channel_num {channel_num}: {unique_val}, count: {count}.')
    return final_df
