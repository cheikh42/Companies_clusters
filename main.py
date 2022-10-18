from utils import siren_list, siren_mapping, tfidf_classification, result_CSV
import numpy as np
import logging 

 # # Basic config ###
logging.basicConfig(
    filename="logger.log", format="%(asctime)s - %(message)s", level=logging.INFO
)

logging.info("Start getting the sirens and companies' names")

list_siren_companies,failed_list = siren_list()

logging.info("End getting the sirens and companies' names")

logging.info("Start clustering companies based on siren numbers")

dict_sirens = siren_mapping(list_siren_companies)

logging.info("End clustering companies based on siren numbers")

logging.info("Start clustering the companies without siren numbers using tf-idf")

dict_failed = tfidf_classification(np.array(failed_list))

logging.info("End clustering the companies without siren numbers using tf-idf")

logging.info("Start creating the result file : 'dataset/result.csv'")

result_CSV(dict_sirens,dict_failed)

logging.info("End creating the result file ")


