
#pr3 -1


import pandas as pd

titanic_df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

print("First few rows of the Titanic dataset:")
print(titanic_df.head())

grouped_stats = titanic_df.groupby('Pclass')['Age'].describe()

print("\nSummary statistics of Age grouped by Pclass:")
print(grouped_stats)

mean_age_by_class = titanic_df.groupby('Pclass')['Age'].mean().tolist()

print("\nMean Age for each Passenger Class:")
print(mean_age_by_class)


"""
#pr3 -2


import seaborn as sns
import pandas as pd

iris = sns.load_dataset('iris')

print("First few rows of the Iris dataset:")
print(iris.head())

setosa_stats = iris[iris['species'] == 'setosa'].describe()

print("\nStatistical details for 'Iris-setosa':")
print(setosa_stats)

versicolor_stats = iris[iris['species'] == 'versicolor'].describe()

print("\nStatistical details for 'Iris-versicolor':")
print(versicolor_stats)

virginica_stats = iris[iris['species'] == 'virginica'].describe()

print("\nStatistical details for 'Iris-virginica':")
print(virginica_stats)

"""