from itertools import cycle
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

# Load the Titanic dataset
titanic = sns.load_dataset('titanic')

styles = {0:'s', 1 : 'X'}
hatches = ['/', 'x']
# Plotting a box plot for distribution of age with respect to each gender and survival
plt.figure(figsize=(12, 6))
sns.boxplot(x='sex', y='age', hue='survived', data=titanic, palette='Set2')

ax=sns.boxplot(x='sex', y='age', hue='survived', data=titanic, palette='Set2')
handles,_ = ax.get_legend_handles_labels()
ax.legend(handles,['dead' ,'survived'])
plt.tight_layout()

for a in ax.axes:
    patches = [patche for patch in ax.patches if type(patche)==mpl.patches.PathPatch]
    h = hatches-(len(patches)//len(hatches)) 
    for patch,hatch in zip(patches,h):
        patch.set_hatch(hatch)
        

plt.title('Distribution of Age by Gender and Survival')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.show()

