import argparse
from typing import List
from datasets import (concatenate_datasets, load_dataset)
from huggingface_hub import login
import requests
import os
import tarfile
from datasets import Dataset
import requests
import json
from dataset_utils import (remove_whitespace, remove_longs, remove_start_numbers, shuffle_apply_percentage)
from huggingface_hub import HfApi
import re

def download_preprocess_bible_dataset(language, url, percentage, seed, keep_datasets):
    with open(f"{language}.txt","wb") as f:
        ds_raw = requests.get(url).content
        f.write(ds_raw)
    
    ds_hf = load_dataset("text", data_files={f"{language}.txt"})
    ds_processed = remove_whitespace(ds_hf["train"],"text")
    ds_processed = shuffle_apply_percentage(ds_processed, percentage, seed)

    if keep_datasets == False:
        os.remove(f"{language}.txt")
    
    return ds_processed

def download_preprocess_leipzig_dataset(language, url, percentage, seed, keep_datasets):
    with open(f"{language}.tar.gz","wb") as f:
        ds_tar = requests.get(url).content
        f.write(ds_tar)
        
    with tarfile.open(f"{language}.tar.gz",'r') as f:
        for members in f.getmembers():
            if 'sentences.txt' in members.name:
                members.name = f"{language}.txt"
                f.extract(members,"")
    os.remove(f"{language}.tar.gz")
    
    ds_hf = load_dataset("text", data_files={f"{language}.txt"})
    ds_processed = ds_hf["train"].map(remove_start_numbers)
    ds_processed = remove_whitespace(ds_processed,"text")    
    ds_processed = shuffle_apply_percentage(ds_processed, percentage, seed)
    
    if keep_datasets == False:
        os.remove(f"{language}.txt")
    
    return ds_processed

def download_preprocess_oscar_dataset(language, url, percentage, seed, token, keep_datasets):
    ds_stream = load_dataset("oscar-corpus/mOSCAR", url[6:], streaming=True, split='train')
    # Get length of mOSCAR subsection
    length = requests.get("https://datasets-server.huggingface.co/info?dataset=oscar-corpus/mOSCAR", headers={"Authorization": f"Bearer {token}"}).json()['dataset_info'][url[6:]]['splits']['train']['num_examples']
    ds = Dataset.from_list(list(ds_stream.take(int((float(percentage)/100)*length))))
    sentences=[]
    for each in ds["text"]:
        each = json.loads(each)
        for every in each:
            sentences.append(every["text"])
    ds_new = Dataset.from_dict({'text':sentences})
    ds_processed = remove_whitespace(ds_new,"text")
    ds_processed = remove_longs(ds_processed,"text")
    ds_processed = ds_processed.shuffle(seed=seed)

    if keep_datasets == True:
        with open(f"{language}.txt","wb") as f:
            f.write(sentences)
    
    return ds_processed

def download_preprocess_glot_dataset(language, url, percentage, seed, token, keep_datasets):
    ds_stream = load_dataset("cis-lmu/Glot500", url[5:], streaming=True, split='train')
    # Get length of glot subsection
    length = requests.get("https://datasets-server.huggingface.co/info?dataset=cis-lmu/Glot500", headers={"Authorization": f"Bearer {token}"}).json()['dataset_info'][url[5:]]['splits']['train']['num_examples']
    ds = Dataset.from_list(list(ds_stream.take(int((float(percentage)/100)*length))))
    sentences=[]
    for each in ds["text"]:
        sentences.append(each)
    ds_new = Dataset.from_dict({'text':sentences})
    ds_processed = remove_whitespace(ds_new,"text")
    ds_processed = remove_longs(ds_processed,"text")
    ds_processed = ds_processed.shuffle(seed=seed)

    if keep_datasets == True:
        with open(f"{language}.txt","wb") as f:
            f.write(sentences)
    
    return ds_processed

def parse_languages(languages):
    percentages = {}
    for index, language in enumerate(languages):
        if re.search("[0-9]", language):
            percentage = language[3:]
            languages[index] = language[:3]
            percentages[languages[index]] = percentage
        else:
            percentages[language] = "100"
    return languages, percentages

def get_dataset(language, url, percentage, seed, token, keep_datasets):
    if 'bible' in url:
        ds_processed = download_preprocess_bible_dataset(language, url, percentage, seed, keep_datasets)
    elif 'leipzig' in url:
        ds_processed = download_preprocess_leipzig_dataset(language, url, percentage, seed, keep_datasets)
    elif 'oscar' in url:
        ds_processed = download_preprocess_oscar_dataset(language, url, percentage, seed, token, keep_datasets)
    elif 'glot' in url:
        ds_processed = download_preprocess_glot_dataset(language, url, percentage, seed, token, keep_datasets)
        
    return ds_processed

def assemble_structure(ds_list, languages, repo_id, concatenate, keep_datasets):
    if concatenate == True:
        ds_assembled = concatenate_datasets(ds_list)
        ds_assembled.push_to_hub(repo_id)
        return
    else:
        api = HfApi()
        for index, ds in enumerate(ds_list):
            ds.to_parquet(f"{languages[index]}.parquet")
            api.upload_file(
                path_or_fileobj=str(languages[index]+".parquet"),
                path_in_repo=f"/{languages[index]}/{languages[index]}.parquet",
                repo_id=repo_id,
                repo_type="dataset",
            )
            if keep_datasets == False:
                os.remove(f"{languages[index]}.parquet")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo_id", type=str, default=".",
                        help="HF repo to upload combined dataset to")
    parser.add_argument("--token", type=str, default=".",
                        help="Token for HF login and API requests")
    parser.add_argument("--languages", type=str, nargs='+', default=["abk", "ahk", "apw", "bod", "cdo", "che", "csy", "dzo",
                                                                     "eus19", "gan", "gwi", "kac", "ksw", "kbd", "lhu", "lus",
                                                                     "mya44", "nan63", "nav", "new", "suz", "wuu", "yue83", "zho2"],
                        help="Languages to combine. If you would like to use only a portion of a language's datset, pass this number as a percentage on the end of the language code, e.g. \"eus10\"")
    parser.add_argument("--keep_text_datasets", action='store_true', default=False,
                        help="Whether or not to keep local text backups of datasets")
    parser.add_argument("--keep_parquet_datasets", action='store_true', default=False,
                        help="Whether or not to keep local parquet backups of datasets")
    parser.add_argument("--seed", type=int, default=1,
                        help="Seed with which each dataset is shuffled")
    parser.add_argument("--concatenate", action='store_true', default=False,
                        help="Whether or not to concatenate datasets, rather than maintain individual splits for each language")
    
    args = parser.parse_args()
    
    languages, percentages = parse_languages(args.languages)

    login(token=args.token, add_to_git_credential=True)
    
    urls = {"abk": "glot-abk_Cyrl",
            "ahk": "glot-ahk_Latn",
            "apw": "https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/apw-apwNT.txt",
            "bod": "glot-bod_Tibt",
            "cdo": "https://downloads.wortschatz-leipzig.de/corpora/cdo_community_2017.tar",
            "che": "glot-che_Cyrl",
            "csy": "glot-csy_Latn",
            "dzo": "glot-dzo_Tibt",
            "eus": "glot-eus_Latn",
            "gan": "https://downloads.wortschatz-leipzig.de/corpora/gan_community_2017.tar",
            "gwi": "https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/gwi-gwiNT.txt",
            "kac": "glot-kac_Latn",
            "ksw": "glot-ksw_Mymr",
            "kbd": "glot-kbd_Cyrl",
            "lhu": "glot-lhu_Latn",
            "lus": "glot-lus_Latn",
            "mya": "glot-mya_Mymr",
            "nan": "glot-nan_Latn",
            "nav": "glot-nav_Latn",
            "new": "glot-new_Deva",
            "suz": "glot-suz_Deva",
            "wuu": "glot-wuu_Hani",
            "yue": "glot-yue_Hani",
            "zho": "glot-zho_Hani"}

    ds_list = []
    for each in languages:
        ds_list.append(get_dataset(each, urls[each], percentages[each], args.seed, args.token, args.keep_text_datasets))
        print(each,"done!")
        
    assemble_structure(ds_list, languages, args.repo_id, args.concatenate, args.keep_parquet_datasets)
    
