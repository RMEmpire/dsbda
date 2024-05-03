import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Download and load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
iris = pd.read_csv(url, header=None, names=column_names)

# 1. List down the features and their types
feature_types = iris.dtypes
print("Features and their types:")
print(feature_types)

# 2. Create a histogram for each feature
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
for i, column in enumerate(iris.drop("class", axis=1).columns):
    sns.histplot(iris[column], kde=True, color="skyblue", ax=axes[i//2, i%2])
    axes[i//2, i%2].set_title(f'Histogram of {column}')
plt.suptitle("Histograms for Each Feature", y=1.02)
plt.tight_layout()
plt.show()

# 3. Create a box plot for each feature
plt.figure(figsize=(12, 8))
sns.boxplot(data=iris.drop("class", axis=1), palette="Set2")
plt.title("Box Plots for Each Feature")
plt.show()

# 4. Compare distributions and identify outliers
plt.figure(figsize=(12, 8))
sns.boxplot(x="class", y="sepal_length", data=iris, hue="class", palette="Set2", dodge=False)
plt.title("Box Plot for Sepal Length by Class")
plt.show()