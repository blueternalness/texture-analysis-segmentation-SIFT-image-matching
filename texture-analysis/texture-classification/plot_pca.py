import pandas as pd
import matplotlib.pyplot as plt

# Load the PCA-reduced data from the CSV file
df = pd.read_csv('train_3d_features.csv')

# Initialize the 3D figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Map the numerical labels to their actual class names and specific colors
classes = {0: 'Blanket', 1: 'Brick', 2: 'Grass', 3: 'Stones'}
colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange'}

# Plot each class iteratively to assign the correct color and label
for label, name in classes.items():
    subset = df[df['Label'] == label]
    ax.scatter(subset['PC1'], 
               subset['PC2'], 
               subset['PC3'], 
               c=colors[label], 
               label=name, 
               s=60, 
               edgecolors='k', 
               depthshade=True)

# Formatting the plot
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('3D PCA Feature Space of Texture Images')
ax.legend()

# Display the interactive plot and save it
plt.savefig('pca_3d_plot.png', dpi=300, bbox_inches='tight')
print("Saved 3D plot to pca_3d_plot.png")
plt.show()