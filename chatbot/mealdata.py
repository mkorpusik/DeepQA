"""
Description: Load the all-food meal descriptions collected via AMT.
Author: Mandy Korpusik
Date: 11/28/16
"""

import sys
import os
import re
import pickle

import spacy.en
nlp = spacy.en.English()

def get_matching_toks(alignment, usda_id, meal):
    matching_tokens = []
    for token_index, token_label in enumerate(alignment):
        if token_label[2:] == usda_id:
            matching_tokens.append(meal[token_index])
    return matching_tokens

def spacy_tokenize(in_str):
    tokens = nlp(in_str)
    return [token.orth_ for token in tokens], [token.lemma_ for token in tokens]



class MealData:

    def __init__(self, dirName):
        """
        Args:
            dirName (string): directory where to load the corpus
        """
        meal_lines = open(dirName + 'allfood_diaries_all.txt').readlines()
        self.meals = []
        alignments = open('alignments_allfood_all_cnn_segmenter').readlines()
        self.usda_vecs = pickle.load(open('/usr/users/korpusik/USDA-encoder-data/models/allfood/allfood_matcher_lowercase_nousdacnn_aligned_vecs_dict', 'rb'), encoding='latin1')
        self.food_IDs = [] # list of food ID lists per meal diary
        self.food_descrips = [] # list of food descriptions per meal diary
        self.single_food_descrips = [] # list of single food descriptions
        self.alignments = [] # list of aligned segments per food item

        # load USDA foods (try encoding foods, decoding meals)
        alignment_index = 0
        for foods in open(dirName + 'allfood_food_IDs_all.txt').readlines():
            food_IDs = []
            food_descrips = []
            alignment = alignments[alignment_index].strip().split()
            meal = meal_lines[alignment_index].strip()
            meal = re.sub(' +', ' ', meal)
            
            # skip meals with diff num tokens than alignments
            meal_tokens, lemmas = spacy_tokenize(meal)
            if len(alignment) != len(meal_tokens):
                print('mismatched lengths', alignment, meal_tokens)
                print(alignment_index, len(alignment), len(meal_tokens))
                alignment_index += 1
                continue
            
            for food in foods.split('\t'):
                tokens = food.split(' ')
                usda_id = tokens[0]
                usda_name = ' '.join(tokens[1:]).replace(',', '').strip()
                usda_name = re.sub(' +', ' ', usda_name)
                usda_name = usda_name.replace('in.', '"')
                food_IDs.append(usda_id)
                food_descrips.append(usda_name)
                self.single_food_descrips.append(usda_name)

                # get tokens predicted as aligned with this particular food
                matching_tokens = get_matching_toks(alignment, usda_id, meal_tokens)
                self.alignments.append(' '.join(matching_tokens))
                
            self.food_IDs.append(food_IDs)
            self.food_descrips.append(food_descrips)
            self.meals.append(meal)
            alignment_index += 1

        print(self.meals[0])
        print(self.food_IDs[0])
        print(self.food_descrips[0])
        print(self.single_food_descrips[0])
        print(self.alignments[0])
        assert len(self.meals) == len(self.food_IDs) == len(self.food_descrips)
        assert len(self.single_food_descrips) == len(self.alignments)

    def getMeals(self):
        return self.meals

    def getFoodIDs(self):
        return self.food_IDs

    def getFoodDescrips(self):
        return self.food_descrips

    def getSingleFoodDescrips(self):
        return self.single_food_descrips
    
    def getAlignments(self):
        return self.alignments

    def getEmbeddings(self):
        # get learned embeddings for each food ID
        self.embeddings = []
        for food_id_list in self.getFoodIDs():
            embedding_list = [self.usda_vecs[food_id] for food_id in food_id_list]
            self.embeddings.append(embedding_list)
        return self.embeddings
