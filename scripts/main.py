import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from preprocess import load_and_preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test, _ = load_and_preprocess()

models = {
    "Logistic Regression": LogisticRegression(solver='lbfgs', max_iter=500, random_state=42),
    "Decision Tree": DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', C=1, gamma='scale', random_state=42, probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

# KMeans separately
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
km = KMeans(n_clusters=len(np.unique(y_train)), random_state=42, n_init=10)
km.fit(X_train_pca)
y_pred_kmeans = km.predict(X_test_pca)
results["KMeans"] = accuracy_score(y_test, y_pred_kmeans)

# Plot model comparison
plt.figure(figsize=(10,6))
plt.bar(results.keys(), results.values(), color='skyblue')
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.xticks(rotation=20)
plt.show()
