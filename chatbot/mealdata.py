"""
Description: Load the all-food meal descriptions collected via AMT.
Author: Mandy Korpusik
Date: 11/28/16
"""

import re

class MealData:

    def __init__(self, dirName):
        """
        Args:
            dirName (string): directory where to load the corpus
        """
        self.meals = open(dirName + 'allfood_diaries.txt').readlines()
        self.food_IDs = [] # list of food ID lists per meal diary
        self.food_descrips = [] # list of food descriptions per meal diary

        # load USDA foods (try encoding foods, decoding meals)
        for foods in open(dirName + 'allfood_food_IDs.txt').readlines():
            food_IDs = []
            food_descrips = []
            for food in foods.split('\t'):
                tokens = food.split(' ')
                usda_id = tokens[0]
                usda_name = ' '.join(tokens[1:]).replace(',', '').strip()
                usda_name = re.sub(' +', ' ', usda_name)
                usda_name = usda_name.replace('in.', '"')
                food_IDs.append(usda_id)
                food_descrips.append(usda_name)
                
            self.food_IDs.append(food_IDs)
            self.food_descrips.append(food_descrips)  

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
