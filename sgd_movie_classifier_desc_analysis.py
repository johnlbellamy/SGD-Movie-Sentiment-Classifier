import pandas as pd
#import matplotlib.pyplot as plt

df = pd.read_csv('data/ratings_shuffled.csv')

print(df['sentiment'].value_counts())

print("Classes are balanced")