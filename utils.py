from tqdm import tqdm
from top2vec import Top2Vec
import numpy as np
import pandas as pd
from decouple import config
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn

BASE_URL = str(config("base_url"))
FILE_PATH = str(config("file_path"))
TRIAL  = str(config("trial"))

def unsupervised_class(
    doc_ids: list, names: list, train_type: str, nb_workers: int
):
    """
    unsupervised_class traines a model which clusters of the names.
    Saves it under the names unsupervised_model.

    Args:
        doc_ids (list): the list of ids (the path + the page number)
        names (list): the list of companies' names
        train_type (str): 'fast_learn', 'learn' or 'deep-learn'
        nb_workers (int): the number of threads used for training

    Returns:
        Top2Vec: a trained model
    """

    model = Top2Vec(
        documents=names, document_ids=doc_ids, speed=train_type, workers=nb_workers
    )

    model.save("models/unsupervised_model")
    
    return model


def tfidf_classification(names : list):
    """tfidf_classification clusters the names using the tf-idf scoring scheme.

    Args:
        names (list): list of companies' names. 

    Returns:
        dict: dict of companies that contain similar names
    """

    ### dictionnary of companies and the companies with similar names.
    res_dict = dict()

    ### initialisation with empty lists
    for name in names :
        res_dict[name] = []

    ### vectorization with tf-idf
    vectorizer = TfidfVectorizer(analyzer="word")
    tfidf_matrix = vectorizer.fit_transform(names)    
    
    ### compute the cosine similarity
    cosine_matrix = awesome_cossim_topn(tfidf_matrix, tfidf_matrix.transpose(), names.size, 0.8)
    coo_matrix = cosine_matrix.tocoo()
    
    with tqdm(
        total=len(coo_matrix.row),
        desc="{:30}".format("detecting similar words"),
    ) as pbar:
        
        ### list of handled keys to avoid duplications
        tmp_keys = []
        
        for i in zip(coo_matrix.row, coo_matrix.col):
            
            if i[1] not in tmp_keys: ### if a values is not amoung the handled keys
                res_dict [names[ i[0] ] ].append( names[i[1]] )
                tmp_keys.append(i[1])
                pbar.update(1)
    
    return res_dict


def result_CSV(dict_sirens : dict, dict_failed : dict ) :
    """result_CSV creates a csv with two columns : the company name and the category ( an int ).

    Args:
        dict_sirens (dict): dictionnary of the french companies spotable by the siren.
        dict_failed (dict): dictinnary of the ones which don't have sirens (mainly the french ones)
    """
    names = []
    categories = []

    tmp_category = 1
    
    with tqdm(
        total=len(dict_sirens.keys()) + len(dict_failed.keys() ) ,
        desc="{:30}".format("detecting similar words"),
    ) as pbar:
        
        ### file the companies with sirens
        for k in dict_sirens.keys() : 
            for name in dict_sirens[k] :
                names.append(name)
                categories.append(tmp_category)
            tmp_category += 1
            pbar.update()
        
        ### file the companies with sirens
        for k in dict_failed.keys() : 
            for name in dict_failed[k] :
                names.append(name)
                categories.append(tmp_category)
            tmp_category += 1
            pbar.update(1)

    ### create a dict of the results
    dict_results = dict()
    dict_results["Company name"] = names
    dict_results["Category"] = categories

    ### create a frame
    result_frame = pd.DataFrame( dict_results )

    ### create a csv
    result_frame.to_csv('dataset/result.csv')


def siren_list() :
    """siren_list generates a list of siren numbers and corresponding companies' names. This is done via French public API.

    Returns:
        tuple(list, list): list of joint sirens and companies, list of companies without sirens.
    """

    ### load the dataset
    company_frame = pd.read_excel(FILE_PATH)
    
    ### transform to a list
    company_list = company_frame["Raw name"].to_list()

    ### clean the lists
    company_list = [str(i.replace("\"", "")) for i in company_list]
    company_list = [str(i.replace("\'", "")) for i in company_list]
    
    ### delete duplications if trial not equal to big
    if TRIAL!="big" :
        company_list = list(set(company_list))

    list_siren_companies = []
    failed_list = []

    ### this is the base url for the get request.

    ### we use tqdm to visualize the progression of the get requests
    with tqdm(
            total=len(company_list),
            desc="{:30}".format("getting the siren number"),
        ) as pbar:

        for c in company_list:

            tmp = c+"&page=1&per_page=1"

            try :
                siren = requests.get(BASE_URL+tmp).json()["results"][0]["siren"]            
                list_siren_companies.append([c, siren])
                pbar.update(1)

            except Exception as e :
                failed_list.append(c)
                pbar.update(1)
    


    return list_siren_companies, failed_list

    
def siren_mapping(list_siren_companies : list) : 
    """siren_mapping creates a dictionnary of siren numbers (keys) and companies wwith these sirens (values).

    Args:
        list_siren_companies (list): list of list of siren and correspondig companies.

    Returns:
        dict: dictionnary of siren numbers and companies.
    """
    
    dict_sirens = dict()### dict of sirens (keys) and the companies having these sirens (values).

    ### initialize the dict with empty lists
    for i in list_siren_companies :
        dict_sirens[i[1]] = []

    ### fil these empty lists
    for i in list_siren_companies :
        dict_sirens[i[1]].append(i[0])

    return dict_sirens







