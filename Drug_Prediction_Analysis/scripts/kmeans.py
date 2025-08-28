import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import mode
from preprocess import load_and_preprocess

X_train, X_test, y_train, y_test, _ = load_and_preprocess()

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

num_clusters = len(np.unique(y_train))
model = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
model.fit(X_train_pca)
y_pred = model.predict(X_test_pca)

# Map clusters to true labels
labels = np.zeros_like(y_pred)
for i in range(num_clusters):
    mask = (y_pred == i)
    if np.sum(mask) > 0:
        labels[mask] = mode(y_test[mask])[0]

print("Accuracy:", accuracy_score(y_test, labels))
print("Confusion Matrix:\n", confusion_matrix(y_test, labels))

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, labels), annot=True, fmt='d', cmap='Blues')
plt.title("KMeans - Confusion Matrix")
plt.show()
