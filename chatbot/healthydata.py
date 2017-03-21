"""
Description: Load the meal descriptions and healthy/unhealthy responses collected via AMT.
Author: Mandy Korpusik
Date: 2/22/17
"""

import sys
import os
import re
import pickle
import csv
import nltk
import numpy as np

#import spacy.en
#nlp = spacy.en.English()


def load_usda_vecs():
    load_model = '/usr/users/korpusik/USDA-encoder-data/models/allfood/allfood_matcher_lowercase_nousdacnn_aligned'
    foods = open(load_model+'_foods').readlines()
    embeds = open(load_model+'_embeddings').readlines()
    usda_vecs = {}
    for food, embed in zip(foods, embeds):
        food = food.strip()
        embed = embed.strip()
        usda_vecs[food] = [np.float64(val) for val in embed.split(' ')]
    #usda_vecs = pickle.load(open(load_model+'_vecs_dict', 'rb'))
    return usda_vecs



class HealthyData:

    class Food:
        def __init__(self, ID, name, energy, protein, fat, chol, sodium, carbs, fiber, sugar):
            self.foodID = ID
            self.name = name
            self.energy = energy
            self.protein = protein
            self.fat = fat
            self.chol = chol
            self.sodium = sodium
            self.carbs = carbs
            self.fiber = fiber
            self.sugar = sugar

    def __init__(self, dirName, usda_vecs, healthy_flag = False, food_context=False):
        """
        Args:
            dirName (string): directory where to load the corpus
        """
        self.healthy_flag = healthy_flag
        self.meals = []
        self.foodEmbeddings = []
        self.responses = []
        self.foods = [] # saves a list of Food objects, for each meal
        self.healthyLabels = [] # 0 for unhealthy, 1 for healthy

        with open(dirName + 'healthybatch1results.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # split different sentences into different data samples
                responses = nltk.sent_tokenize(row['Answer.description1'])
                label = row['Answer.selected']

                # get three eaten foods, with nutrients
                foodList = []
                foodIDs = []
                for itemNum in ['1', '2', '3']:
                    foodID = row['Input.FoodID'+itemNum]
                    name = row['Input.foodName'+itemNum]
                    energy = row['Input.energy'+itemNum]
                    protein = row['Input.protein'+itemNum]
                    fat = row['Input.fat'+itemNum]
                    chol = row['Input.chol'+itemNum]
                    sodium = row['Input.sodium'+itemNum]
                    carbs = row['Input.carbs'+itemNum]
                    fiber = row['Input.fiber'+itemNum]
                    sugar = row['Input.sugars'+itemNum]
                    food = self.Food(foodID, name, energy, protein, fat, chol, sodium, carbs, fiber, sugar)
                    foodList.append(food)
                    foodIDs.append(foodID)

                if food_context:
                    if foodIDs[0] not in self.usda_vecs:
                        print('skip unk food', foodIDs[0])
                        continue
                    elif foodIDs[1] not in self.usda_vecs:
                        print('skip unk food', foodIDs[1])
                        continue
                    elif foodIDs[2] not in self.usda_vecs:
                        print('skip unk food', foodIDs[2])
                        continue
                    embeddingSum = np.sum([self.usda_vecs[foodID] for foodID in foodIDs], axis=0)
                    self.foodEmbeddings.append(embeddingSum)

                # add data examples
                for response in responses:
                    self.responses.append(response)
                    self.meals.append(row['Input.meal_response'])
                    self.healthyLabels.append(0 if label=="$(unhealthy)" else 1)
                    self.foods.append(foodList)

        print(self.meals[:2])
        print(self.responses[:2])
        print(self.healthyLabels[:2])
        print([food.foodID for food in self.foods[0]])
        print(self.foodEmbeddings[-1], len(self.foodEmbeddings[-1]))
        assert len(self.meals) == len(self.responses)

    # TODO: append vector of features indicating nutrition facts
    
    def getMeals(self):
        if not self.healthy_flag:
            return self.meals
        else:
            # append healthy/unhealthy at the end of the meal
            meals_plus_healthy = [meal + " healthy" if label else meal + " unhealthy" for (meal, label) in zip(self.meals, self.healthyLabels)]
            print(meals_plus_healthy[0])
            return meals_plus_healthy

    def getResponses(self):
        return self.responses

    def getFoods(self):
        return self.foods

    def getFoodEmb(self):
        return self.foodEmbeddings

    def getFoodIDs(self):
        return [' '.join([food.foodID for food in foodList]) for foodList in self.foods]

    def getLabels(self):
        return self.healthyLabels
