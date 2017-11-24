import os
from os.path import isfile, join
import csv
import imageio
import numpy as np


def get_classes():
    classes = [
        'affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale', 'american_staffordshire_terrier',
        'appenzeller', 'australian_terrier', 'basenji', 'basset', 'beagle', 'bedlington_terrier',
        'bernese_mountain_dog', 'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound', 'bluetick',
        'border_collie', 'border_terrier', 'borzoi', 'boston_bull', 'bouvier_des_flandres', 'boxer',
        'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff', 'cairn', 'cardigan',
        'chesapeake_bay_retriever', 'chihuahua', 'chow', 'clumber', 'cocker_spaniel', 'collie',
        'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo', 'doberman', 'english_foxhound',
        'english_setter', 'english_springer', 'entlebucher', 'eskimo_dog', 'flat-coated_retriever',
        'french_bulldog', 'german_shepherd', 'german_short-haired_pointer', 'giant_schnauzer', 'golden_retriever',
        'gordon_setter', 'great_dane', 'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael',
        'ibizan_hound', 'irish_setter', 'irish_terrier', 'irish_water_spaniel', 'irish_wolfhound',
        'italian_greyhound', 'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier', 'komondor', 'kuvasz',
        'labrador_retriever', 'lakeland_terrier', 'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog',
        'mexican_hairless', 'miniature_pinscher', 'miniature_poodle', 'miniature_schnauzer', 'newfoundland',
        'norfolk_terrier', 'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog', 'otterhound',
        'papillon', 'pekinese', 'pembroke', 'pomeranian', 'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler',
        'saint_bernard', 'saluki', 'samoyed', 'schipperke', 'scotch_terrier', 'scottish_deerhound',
        'sealyham_terrier', 'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier',
        'soft-coated_wheaten_terrier', 'staffordshire_bullterrier', 'standard_poodle', 'standard_schnauzer',
        'sussex_spaniel', 'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier', 'vizsla',
        'walker_hound', 'weimaraner', 'welsh_springer_spaniel', 'west_highland_white_terrier', 'whippet',
        'wire-haired_fox_terrier', 'yorkshire_terrier'
    ]
    return {classes[i]: i for i in range(len(classes))}


def get_training_data(num_examples=-1, positive=1, negative=-1):
    classes = get_classes()
    labels = dict()
    with open('data/labels.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            labels[row[0]] = row[1]

    data_path = 'data/train/'
    files = [f for f in os.listdir(data_path) if isfile(join(data_path, f))]
    files = files[:num_examples]

    input_data = [np.array([]) for _ in files]
    output_data = [np.array([]) for _ in files]
    for i, f in enumerate(files):
        input_data[i] = imageio.imread(join(data_path, f))
        output_data[i] = negative * np.ones((1, len(classes)))
        output_data[i][0, classes[labels[f.split('.')[0]]]] = positive
    return np.array(input_data), np.array(output_data)


def main():
    input_data, output_data = get_training_data(num_examples=1)
    print input_data, output_data


if __name__ == '__main__':
    main()
