# %%
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

OUTPUT_DIR = "Outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

iris = load_iris()
x = iris.data
y = iris.target
print(iris.feature_names, iris.target_names) 

# %%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# %%
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)

# %%
y_pred = model.predict(x_test)
print("Predictions:", y_pred[:5])
print("True labels:", y_test[:5])

# %%
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# %%
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(x_train, y_train)
y_pred2 = model2.predict(x_test)
print("k-NN accuracy:", accuracy_score(y_test, y_pred2))

# %%
model_path = os.path.join(OUTPUT_DIR, "decision_tree_model.pkl")
joblib.dump(model, model_path)
print(f"Model saved to: {model_path}")

# %%
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap="Blues",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Save confusion matrix image
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()

print(f"Confusion matrix saved to: {cm_path}")


