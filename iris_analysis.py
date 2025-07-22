
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame
df['species'] = df['target'].map(dict(enumerate(iris.target_names)))

# Exploration
print(df.head())
print(df.info())
print(df.isnull().sum())

# Basic statistics
print(df.describe())

# Grouped means
print(df.groupby('species').mean(numeric_only=True))

# Visualizations
plt.figure()
plt.plot(df.index, df['sepal length (cm)'])
plt.title('Simulated Time-Series of Sepal Length')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')
plt.savefig('line_chart.png')

plt.figure()
sns.barplot(x='species', y='petal length (cm)', data=df, ci=None)
plt.title('Average Petal Length by Species')
plt.savefig('bar_chart.png')

plt.figure()
plt.hist(df['sepal width (cm)'], bins=20)
plt.title('Distribution of Sepal Width')
plt.savefig('histogram.png')

plt.figure()
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title('Sepal Length vs Petal Length')
plt.savefig('scatter_plot.png')
