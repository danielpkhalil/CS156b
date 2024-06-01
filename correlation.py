import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/groups/CS156b/data/student_labels/train2023.csv')

filtered_df = df.iloc[:, 7:16] 

correlation_matrix = filtered_df.corr()

print(correlation_matrix)

plt.figure(figsize=(14, 12))
ax = sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                 square=True, linewidths=.5, cbar_kws={"shrink": .8})

plt.title('Correlation Matrix of Medical Attributes', fontsize=16)  # Larger title font
ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=12)  # Larger x-axis labels
ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=12)  # Larger y-axis labels

# Show the plot
plt.show()
