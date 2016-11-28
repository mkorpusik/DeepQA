"""
Description: Load the all-food meal descriptions collected via AMT.
Author: Mandy Korpusik
Date: 11/28/16
"""

class MealData:

    def __init__(self, dirName):
        """
        Args:
            dirName (string): directory where to load the corpus
        """
        self.meals = open(dirName + 'allfood_diaries.txt').readlines()

        # TODO: load USDA foods (try encoding foods, decoding meals)

    def getMeals(self):
        return self.meals
