'''
Miscellaneous utility functions
'''
import collections
import operator
import os
import time
import datetime
import shutil

import pickle
import numpy as np

def order_dictionary(dictionary, mode, reverse=False):
    '''
    Order a dictionary by 'key' or 'value'.
    mode should be either 'key' or 'value'
    http://stackoverflow.com/questions/613183/sort-a-python-dictionary-by-value
    '''

    if mode =='key':
        return collections.OrderedDict(sorted(dictionary.items(),
                                              key=operator.itemgetter(0),
                                              reverse=reverse))
    elif mode =='value':
        return collections.OrderedDict(sorted(dictionary.items(),
                                              key=operator.itemgetter(1),
                                              reverse=reverse))
    elif mode =='key_value':
        return collections.OrderedDict(sorted(dictionary.items(),
                                              reverse=reverse))
    elif mode =='value_key':
        return collections.OrderedDict(sorted(dictionary.items(),
                                              key=lambda x: (x[1], x[0]),
                                              reverse=reverse))
    else:
        raise ValueError("Unknown mode. Should be 'key' or 'value'")

def reverse_dictionary(dictionary):
    '''
    http://stackoverflow.com/questions/483666/python-reverse-inverse-a-mapping
    http://stackoverflow.com/questions/25480089/right-way-to-initialize-an-ordereddict-using-its-constructor-such-that-it-retain
    '''
    #print('type(dictionary): {0}'.format(type(dictionary)))
    if type(dictionary) is collections.OrderedDict:
        #print(type(dictionary))
        return collections.OrderedDict([(v, k) for k, v in dictionary.items()])
    else:
        return {v: k for k, v in dictionary.items()}

def merge_dictionaries(*dict_args):
    '''
    http://stackoverflow.com/questions/38987/how-can-i-merge-two-python-dictionaries-in-a-single-expression
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def get_basename_without_extension(filepath):
    '''
    Getting the basename of the filepath without the extension
    E.g. 'data/formatted/movie_reviews.pickle' -> 'movie_reviews'
    '''
    return os.path.basename(os.path.splitext(filepath)[0])

def create_folder_if_not_exists(directory):
    '''
    Create the folder if it doesn't exist already.
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_current_milliseconds():
    '''
    http://stackoverflow.com/questions/5998245/get-current-time-in-milliseconds-in-python
    '''
    return(int(round(time.time() * 1000)))


def get_current_time_in_seconds():
    '''
    http://stackoverflow.com/questions/415511/how-to-get-current-time-in-python
    '''
    return(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))

def get_current_time_in_miliseconds():
    '''
    http://stackoverflow.com/questions/5998245/get-current-time-in-milliseconds-in-python
    '''
    return(get_current_time_in_seconds() + '-' + str(datetime.datetime.now().microsecond))


def convert_configparser_to_dictionary(config):
    '''
    http://stackoverflow.com/questions/1773793/convert-configparser-items-to-dictionary
    '''
    my_config_parser_dict = {s:dict(config.items(s)) for s in config.sections()}
    return my_config_parser_dict


def copytree(src, dst, symlinks=False, ignore=None):
    '''
    http://stackoverflow.com/questions/1868714/how-do-i-copy-an-entire-directory-of-files-into-an-existing-directory-using-pyth
    '''
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def load_pickle(file_path):
    print(file_path)
    with open(file_path, "rb") as file:
        return pickle.load(file)

def pad_list(old_list, padding_size, padding_value):
    '''
    http://stackoverflow.com/questions/3438756/some-built-in-to-pad-a-list-in-python
    Example: pad_list([6,2,3], 5, 0) returns [6,2,3,0,0]
    '''
    assert padding_size >= len(old_list), "padding_size = " + str(padding_size) + "  len(old_list) = " + str(len(old_list))
    return list(old_list) + [padding_value] * (padding_size-len(old_list))

def pad_batch(dataset, sequence_number, dataset_type):
    batch = {}

    batch_token_indices_sequence = np.array(dataset.token_indices[dataset_type])[sequence_number]
    batch['sequence_lengths'] = np.array(dataset.sequence_lengths[dataset_type])[sequence_number]
    max_sequence_lengths = max(np.array(dataset.sequence_lengths[dataset_type])[sequence_number])
    batch['token_indices_padded'] = [pad_list(token_indices_sequence, max_sequence_lengths, dataset.PADDING_TOKEN_INDEX)
                                         for token_indices_sequence in batch_token_indices_sequence]

    label_vector_indices = np.array(dataset.label_vector_indices[dataset_type])[sequence_number]
    batch['label_vector_indices'] = [pad_list(label_vector, max_sequence_lengths, dataset.PADDING_LABEL_VECTOR)
                                         for label_vector in label_vector_indices]
    label_indices = np.array(dataset.label_indices[dataset_type])[sequence_number]
    batch['label_indices'] = [pad_list(label, max_sequence_lengths, dataset.PADDING_LABEL_INDEX)
                                     for label in label_indices]

    batch_character_indices = np.array(dataset.character_indices[dataset_type])[sequence_number]
    longest_token_length_in_sequence = max(np.array(dataset.longest_token_length_in_sequence[dataset_type])[sequence_number])
    character_indices = [[pad_list(temp_token_indices, longest_token_length_in_sequence, dataset.PADDING_CHARACTER_INDEX)
                                                        for temp_token_indices in character_indices ] for character_indices in batch_character_indices]

    batch['character_indices_padded'] = [pad_list(character_indice, max_sequence_lengths, [0] * longest_token_length_in_sequence)
                                     for character_indice in character_indices]

    batch_token_lengths = np.array(dataset.token_lengths[dataset_type])[sequence_number]
    batch_token_lengths = [pad_list(token_length, max_sequence_lengths, 0)
      for token_length in batch_token_lengths]
    batch['token_lengths'] = batch_token_lengths

    gazetteer_indices = dataset.gazetteer_indices[dataset_type]
    if gazetteer_indices:
        gazetteer_indices = np.array(gazetteer_indices)[sequence_number]
        batch['gazetteer_indices'] = np.array([pad_list(item, max_sequence_lengths, 0) for item in gazetteer_indices])

    else:
        batch['gazetteer_indices'] = np.zeros((len(sequence_number), max_sequence_lengths))

    return batch