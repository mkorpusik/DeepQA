"""
Description: Load the meal descriptions and healthy/unhealthy responses collected via AMT.
Author: Mandy Korpusik
Date: 2/22/17
"""

import sys
import os
import re
import pickle
import json
import csv
import nltk
import numpy as np
import string
import xlrd
import queue
import random


def find_neighbor(food, vec, usda_vecs):
    neighbors_q = queue.PriorityQueue()
    usda_foods = set() # prevent repeats
    if type(usda_vecs) == dict:
        usda_vecs = usda_vecs.items()
    for usda_food, usda_vec in usda_vecs:
        if food == usda_food or usda_food == 0:
            continue
        dist = nltk.cluster.util.euclidean_distance(np.array(vec), np.array(usda_vec))
        #print(usda_food)
        if ' '.join(usda_food) not in usda_foods:
            neighbors_q.put((dist, usda_food))
            usda_foods.add(' '.join(usda_food))
    return neighbors_q



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

    def __init__(self, dirName, usda_vecs, healthy_flag = False, augment = False, motivate_only = False, advice_only = False):
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
        self.usda = json.load(open('/usr/users/korpusik/USDA-encoder-data/models/allfood/allfood_matcher_lowercase_nousdacnn_aligned_usda'))

        # TODO: append vector of features indicating nutrition facts

        files = ['salad1.csv', 'salad2.csv', 'salad3.csv', 'dinner1.csv', 'dinner2.csv', 'dinner3.csv', 'pasta1.csv', 'pasta2.csv', 'pasta3.csv', 'pasta4.csv', 'healthybatch1results.xls', 'moreEncouragingResponses1.xls', 'healthyfeedbackattempt1results_encouraging.xls']

        count = 0
        for filen in files:
            # process excel file to match csv
            if filen[-3:] == 'xls':
                book = xlrd.open_workbook(dirName + filen)
                sheet = book.sheet_by_name(book.sheet_names()[0])
                labels = sheet.row_values(0)
                reader = [sheet.row_values(idx) for idx in range(1, sheet.nrows)]
            else:
                csvfile = open(dirName + filen)
                reader = csv.DictReader(csvfile)
            for row in reader:
                count += 1
                # skip every 10th line (for testing only)
                if count % 10 == 0 and filen[-3:] != 'xls':
                    print('skipping test sent', row['Input.meal_response'])
                    continue

                # split different sentences into different data samples
                if filen[-3:] == 'xls':
                    meal = row[labels.index('Input.meal_response')]
                    if advice_only:
                        responses = []
                    else:
                        responses = nltk.sent_tokenize(row[labels.index('Answer.description1')])
                else:
                    meal = row['Input.meal_response']
                    if advice_only:
                        responses = []
                    else:
                        responses = nltk.sent_tokenize(row['Answer.description1'])
                #print(meal)

                # check if Turker wrote two different responses
                if filen[-3:] == 'xls' and not motivate_only:
                    if 'Answer.description2' in labels:
                        responses.extend(nltk.sent_tokenize(row[labels.index('Answer.description2')]))
                elif not motivate_only:
                    if 'Answer.description2' in row:
                        responses.extend(nltk.sent_tokenize(row['Answer.description2']))

                # add responses w/o punctuation (keeping exclamation points)
                new_responses = []
                for response in responses:
                    new_responses.append(response)
                    new_responses.append(''.join(ch for ch in response if (ch not in string.punctuation or ch== '!')))
                responses = new_responses

                if filen[-3:] == 'xls':
                    label = row[labels.index('Answer.selected')]
                else:
                    label = row['Answer.selected']

                # get three eaten foods, with nutrients
                foodList = []
                foodIDs = []
                neighborIDs = [] # nearest neighbor to each food
                for itemNum in ['1', '2', '3']:
                    if filen[-3:] == 'xls':
                        foodID = row[labels.index('Input.FoodID'+itemNum)]
                        foodID = str(int(foodID))
                        name = row[labels.index('Input.foodName'+itemNum)]
                        energy = row[labels.index('Input.energy'+itemNum)]
                        protein = row[labels.index('Input.protein'+itemNum)]
                        fat = row[labels.index('Input.fat'+itemNum)]
                        chol = row[labels.index('Input.chol'+itemNum)]
                        sodium = row[labels.index('Input.sodium'+itemNum)]
                        carbs = row[labels.index('Input.carbs'+itemNum)]
                        fiber = row[labels.index('Input.fiber'+itemNum)]
                        sugar = row[labels.index('Input.sugars'+itemNum)]
                    else:
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

                    if augment:
                        print(name)
                        if foodID not in usda_vecs:
                            print('skip unk food', foodIDs[0])
                            continue
                        neighbors_q = find_neighbor(foodID, usda_vecs[foodID], usda_vecs)
                        neighbor = neighbors_q.get()[1]
                        neighborIDs.append(neighbor)
                        print( neighbor, self.usda[neighbor] )

                if foodIDs[0] not in usda_vecs:
                    print('skip unk food', foodIDs[0])
                    continue
                elif foodIDs[1] not in usda_vecs:
                    print('skip unk food', foodIDs[1])
                    continue
                elif foodIDs[2] not in usda_vecs:
                    print('skip unk food', foodIDs[2])
                    continue
                embeddingSum = np.sum([usda_vecs[foodID] for foodID in foodIDs], axis=0)

                # add data examples
                for response in responses:
                    self.responses.append(response)
                    self.meals.append(meal)
                    self.healthyLabels.append(0 if label=="$(unhealthy)" else 1)
                    self.foods.append(foodList)
                    self.foodEmbeddings.append(embeddingSum)

                # add example with neighbor embedding sum instead
                if augment:
                    neighborEmbeddingSum = np.sum([usda_vecs[foodID] for foodID in neighborIDs], axis=0)
                    for response in responses:
                        self.responses.append(response)
                        self.meals.append(meal)
                        self.healthyLabels.append(0 if label=="$(unhealthy)" else 1)
                        self.foods.append(foodList)
                        self.foodEmbeddings.append(neighborEmbeddingSum)

        print(self.meals[:2])
        print(self.responses[:2])
        print(self.healthyLabels[:2])
        print([food.foodID for food in self.foods[0]])
        print(self.foodEmbeddings[-1], len(self.foodEmbeddings[-1]))
        assert len(self.meals) == len(self.responses) == len(self.foods)
        assert len(self.foods) == len(self.healthyLabels) == len(self.foodEmbeddings)
        print(len(self.meals))

    
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

    def getWords(self):
        wordList = []
        for response in self.getResponses():
            wordList.append("<start>")
            wordList.extend(nltk.tokenize.wordpunct_tokenize(response))
        return wordList

    def getFoods(self):
        return self.foods

    def getFoodEmb(self):
        return self.foodEmbeddings

    def getFoodIDs(self):
        return [' '.join([food.foodID for food in foodList]) for foodList in self.foods]

    def getLabels(self):
        return self.healthyLabels
