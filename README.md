# Company Clustering

This project aims at cluster companies' names to the same legal entity or group.
First, it is advised to work in a virtual environment. To set it up here is the procedure :
1. Install virtualenv using pip [Virtual env with pip](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/):
    ```
    python3 -m pip install --user virtualenv
    ```
2. Create a virtual environement - venv :
     ```
    virtualenv venv
    ```
3. Activate venv :
     ```
    venv/bin/activate
    ```
4. Install the requirements in requirements.txt :
     ```
    pip install -r requirements.txt 
    ```
5. Once finished with venv deactivate it :
     ```
    venv/bin/deactivate
    ```
6. And delete it if you want.

This allows you to avoid all the needed libraries on your computer.

There are several files and folders in this project:

## requirements.txt
This files contains the libraries that are used in this project :
- numpy : mainly used here to store and access lists and dictionnary,
- pandas : used to manipulate csv and excel files,
- tqdm : to follow the progression in a loop,
- wordcloud : to create a cloud of words in images - as in the notebook,
- matplotlib : to visualize wordcloud images,
- black : to format our code,
- top2vec : to cluster the companies name in an usupervised manner,
- joblib : used by top2vec,
- scikit-learn : to used tf-idf vectorization,
- sparse_dot_topn : to do sparse matrices multiplications faster than the other libraries,
- python-decouple : to access environmental variables,
- cython : used by sparse_dot_topn.


## .env
contains the environmenta variables : 
- base_url : the base url to the siren API,
- file_path : the path towards the excel file containing the companies names,
- trial : equal to big if we are handlening the big excel file example.

## notebook.ipynb
This notebook explores the data and the API [Recherche entreprise](https://api.gouv.fr/documentation/api-recherche-entreprises) used to spot siren number for French companies. As well as the clusering using ML techniques - used for the non-French companies which aren't handled by this API. 
In this notebook we also explain the procedure to handle bigger file (100,000 rows).

## utils.py
This script contains the functions that are used in other files. All of these functions contain docstrings - Which allows us to see its utility without investigating the code. 

## main.py
This script runs all the process that leads to a **CSV** result file : 
1. It parses the specified file (**file_path in the .env**).
2. Clusters the documents using the siren numbers. 
3. Clusters the documents that do not have siren numbers using the tf-idf metric. 

to run it use the following : 
```
python main.py
```
## dataset 
This folder contains the temporary npy files (dicts and lists), the excel file containing the companies' names and, the result files (**result.csv**). 
**result.csv** contains two row :  *Company name* and *Categories*. *Categories* is an integer that gather the similar companies/ companies with the same siren number.

## models
This folder contains trained Top2Vec model. Which is negelected because the clustering isn't relevant (10 categories out of 5000).


