import pandas as pd
import json
import os
from tqdm import tqdm
import sys
from langdetect import detect
import re
import pickle
import string

def read_files():

    # Creates a dataframe with the needed columns
    df = pd.DataFrame(columns=['paper_id', 'title', 'abstract', 'text'])

    # Path to the directories with the json
    bioarxiv_dir = 'Data/biorxiv_medrxiv/biorxiv_medrxiv'
    comm_use_dir = 'Data/comm_use_subset/comm_use_subset'
    custom_license_dir = 'Data/custom_license/custom_license'
    noncomm_use_dir = 'Data/noncomm_use_subset/noncomm_use_subset'

    data_directories = [bioarxiv_dir, comm_use_dir, custom_license_dir, noncomm_use_dir]

    num_documents = 0
    print('Reading files...\n')
    for dir in data_directories:
        print('Reading {} files'.format(dir))
        for filename in tqdm(os.listdir(dir)):
            filename = os.path.join(dir, filename)
            with open(filename, 'rb') as f:
                row = _read_file(f)
                if row is None:
                    continue
                df = df.append(row, ignore_index=True)
            num_documents += 1

    print('\nThe dataset consists of {} documents'.format(num_documents))

    # Calculate the size of the dataframe
    dec = sys.getsizeof(df) % 1000000
    df_size = sys.getsizeof(df)//1000000

    print('The size of the dataframe is of {},{} MB'.format(df_size, dec))

    return df


def _read_file(json_file):
    """
    Function that reads a json file and outputs a dict to append to a df.
    It doesn't take a document into account if:
        - Not written in English
        - Number of words in text < 500.
    If there is no abstract, it takes the title as the abstract
    TODO Think if we want the authors too
    :param json_file: path to the jsonfile
    :return: dict with keys: [paper_id, title, abstract, text]
    """
    # Load json file
    file_dict = json.load(json_file)

    # Get information
    paper_id = file_dict['paper_id']
    title = file_dict['metadata']['title'] # string
    abstract = '\n'.join([paragraph['text'] for paragraph in file_dict['abstract']]) # string

    # Check if the abstract is written in English
    if type(abstract) == str and len(abstract) > 10:
        if detect(abstract) != 'en':
            return None
    else:
        abstract = title

    text = [paragraph['text'] for paragraph in file_dict['body_text']]
    text = '\n'.join(text) # uncomment this line if instead of list of paragraph you want the whole text as a string

    # If there are less than 500 words, don't consider the document
    if len(re.findall(r"[\w']+|[.,!?;]", text)) < 500:
        return None

    # If there is no abstract, check if text is written in English
    if type(abstract) != str:
        if detect(text) != 'en':
            return None

    return {'paper_id': paper_id, 'title': title, 'abstract': abstract, 'text': text}


def save_dictionaries(df):
    """
    Calculates the dictionaries needed to compute TFIDF score.
        term_frequencies: list of dicts of term frequencies within a document.
        document_frequencies: number of documents containing a given term.
        document_length
    The dictionaries will be stored in the directory Data/ranking_dict/
    """
    directory = 'Data/ranking_dict/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    # CREATE TEXT DICTIONARIES

    term_frequencies = {}  # dict of dicts paper_id -> word -> frequency within a document
    document_frequencies = {}  # dict word -> number of documents containing the term
    document_length = {}  # dict paper_id -> document length

    for id in list(df.paper_id):
        term_frequencies[id] = {}
        document_length[id] = 0

    print('Processing the corpus...\n')
    for id, document in tqdm(zip(list(df.paper_id), list(df.text))):
        actual_frequencies = {}
        words_set = set()
        length = 0
        if type(document) != type('a'):
            continue
        for word in re.findall(r"[\w']+|[.,!?;]", document.strip()):
            word = word.lower()
            if word in string.punctuation:
                continue
            length += 1
            if word in actual_frequencies:
                actual_frequencies[word] += 1
            else:
                actual_frequencies[word] = 1
            words_set.add(word)
        for word in words_set:
            if word in document_frequencies:
                document_frequencies[word] += 1
            else:
                document_frequencies[word] = 1
        document_length[id] = length
        term_frequencies[id] = actual_frequencies

    # Save dictionaries into files
    with open('Data/ranking_dict/document_frequencies_text.p', 'wb') as fp:
        pickle.dump(document_frequencies, fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open('Data/ranking_dict/term_frequencies_text.p', 'wb') as fp:
        pickle.dump(term_frequencies, fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open('Data/ranking_dict/document_length_text.p', 'wb') as fp:
        pickle.dump(document_length, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print(len(term_frequencies))
    # REPEAT WITH ABSTRACTS

    term_frequencies = {}  # dict of dicts paper_id -> word -> frequency within a document
    document_frequencies = {}  # dict word -> number of documents containing the term
    document_length = {}  # dict paper_id -> document length

    for id in list(df.paper_id):
        term_frequencies[id] = {}
        document_length[id] = 0

    print('Processing the corpus...\n')
    for id, document in tqdm(zip(list(df.paper_id), list(df.abstract))):
        actual_frequencies = {}
        words_set = set()
        length = 0
        if type(document) != type('a'):
            continue
        for word in re.findall(r"[\w']+|[.,!?;]", document.strip()):
            word = word.lower()
            if word in string.punctuation:
                continue
            length += 1
            if word in actual_frequencies:
                actual_frequencies[word] += 1
            else:
                actual_frequencies[word] = 1
            words_set.add(word)
        for word in words_set:
            if word in document_frequencies:
                document_frequencies[word] += 1
            else:
                document_frequencies[word] = 1
        document_length[id] = length
        term_frequencies[id] = actual_frequencies

    # Save dictionaries into files
    with open('Data/ranking_dict/document_frequencies_abstract.p', 'wb') as fp:
        pickle.dump(document_frequencies, fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open('Data/ranking_dict/term_frequencies_abstract.p', 'wb') as fp:
        pickle.dump(term_frequencies, fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open('Data/ranking_dict/document_length_abstract.p', 'wb') as fp:
        pickle.dump(document_length, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print(len(term_frequencies))