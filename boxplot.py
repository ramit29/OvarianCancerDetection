import seaborn as sns
import pandas as pd
#df = sns.load_dataset('tips')
import matplotlib.pyplot as plt
df = pd.read_csv("Master1-4.csv")

matrix = df.drop(['ID', 'Class','Location','Batch'], axis = 1)
matrix= matrix.var(axis=1)
print(matrix)
df['Variance']=matrix
# Grouped boxplot
sns.boxplot(x="Class", y='Variance', hue="Batch", data=df, palette="Set1")
plt.show()
