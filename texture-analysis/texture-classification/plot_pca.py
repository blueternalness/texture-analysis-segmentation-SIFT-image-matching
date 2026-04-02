import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('train_3d_features.csv')
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
classes = {0: 'Blanket', 1: 'Brick', 2: 'Grass', 3: 'Stones'}
colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange'}
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

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('3D PCA Feature Space of Texture Images')
ax.legend()

plt.savefig('pca_3d_plot.png', dpi=300, bbox_inches='tight')
print("Saved 3D plot to pca_3d_plot.png")
plt.show()