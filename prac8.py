import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
titanic = sns.load_dataset('titanic')

# 1. Visualizing patterns in the data
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

styles = {0:'s', 1 : 'X'}

# Use Seaborn's countplot to visualize the number of passengers in each class
sns.countplot(x='class', hue='survived', data=titanic, palette='Set1', hatch=['//', '*'])
ax = sns.scatterplot(x='class',hue='survived', data=titanic, style='survived', markers=styles)
ax.legend(title='Survived', bbox_to_anchor=(1.05,1), labels=['dead' ,'not dead'])
plt.tight_layout()
bars = plt.gca().patches
count = 1
for b in bars: 
    if count  in range(4):
        count+=1
        
        
    else:
        count+=1
        b.set_hatch("*")
        

    
plt.title('Survival Count by Passenger Class')
plt.show()

# 2. Plotting a histogram for the distribution of ticket prices (fare)
plt.figure(figsize=(12, 6))

# Use Seaborn's histogram to visualize the distribution of ticket prices
sns.histplot(titanic['fare'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Ticket Prices (Fare)')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()
