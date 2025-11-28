from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PCA import FromScratchPCA


data = pd.read_csv("./faces.csv", header=None).to_numpy()

n_components = 250

model1 = PCA(n_components).fit(data)

model2 = FromScratchPCA(n_components).fit(data)


recons_faces_1 = model1.inverse_transform(model1.transform(data))
recons_faces_2 = model2.inverse_transform(model2.transform(data))

plt.figure(figsize=(10,5))
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, 
                    wspace=0.3, hspace=0.4)

for i in range(3):
    for j in range(4):
        idx = np.random.randint(len(data))
        
        plt.subplot(3,8, 8*i + 2*j + 1 )
        plt.imshow(recons_faces_1[idx].reshape(80,70), cmap="gray")
        plt.title("Scikit-Learn")        
        plt.axis("off")
        
        plt.subplot(3,8, 8*i + 2*j + 2 )
        plt.imshow(recons_faces_2[idx].reshape(80,70), cmap="gray")
        plt.title("Implemented")        
        plt.axis("off")

plt.show()