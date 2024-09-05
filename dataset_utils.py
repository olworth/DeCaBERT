import re
from datasets import (concatenate_datasets, load_dataset)

def remove_whitespace(ds,col_name):
    safe_indices = []
    index = 0
    for each in ds[col_name]:
        if each != '' and each != ' ' and each!= '<range>':
            safe_indices.append(index)
        index+=1

    ds_whitespaced = ds.select(safe_indices)
    
    return ds_whitespaced

def remove_longs(ds,col_name):
    safe_indices = []
    index = 0
    for each in ds[col_name]:
        if len(each) <= 1000:
            safe_indices.append(index)
        index+=1

    ds_shorted = ds.select(safe_indices)
    
    return ds_shorted

def remove_start_numbers(data):
    while re.search("[0-9]", data["text"][0]):
        data["text"]=data["text"][1:]

    return data

def shuffle_apply_percentage(ds, percentage, seed):
    ds = ds.shuffle(seed=seed)
    ds = ds.select(range(0,int((float(percentage)/100)*len(ds))))
    return ds
