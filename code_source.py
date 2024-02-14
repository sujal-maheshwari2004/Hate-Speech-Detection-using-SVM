
# Importing necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Importing SVM
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
data = pd.read_csv("data.csv")

# Mapping labels for better readability
data["labels"] = data["class"].map({0: "Hate Speech detected", 1: "Offensive Speech detected", 2: "No Hate and Offensive Speech detected"})

# Selecting only relevant columns
data = data[["tweet", "labels"]]

# Cleaning the text data
def clean(text):
    # your cleaning function here
    return text

data["tweet"] = data["tweet"].apply(clean)

# Display class distribution pie chart
class_distribution = data["labels"].value_counts()
class_distribution.plot(kind='pie', autopct='%1.1f%%', startangle=90, explode=(0.1, 0, 0), colors=['#ff9999', '#66b3ff', '#99ff99'])
plt.title('Class Distribution')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(data["tweet"], data["labels"], test_size=0.2, random_state=42)

# Convert text data to numerical representation using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Model building - Support Vector Machine (SVM)
model = SVC(kernel='linear', C=1.0)  # Using SVM

# Training the model
model.fit(X_train_vectorized, y_train)

# Testing the model
y_pred = model.predict(X_test_vectorized)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}\n')

# Precision, F1 score, and classification report
precision = precision_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print('Classification Report:')
print(classification_report(y_test, y_pred))

print('\nConfusion Matrix:')
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

print(f'Precision: {precision:.2f}')
print(f'F1 Score: {f1:.2f}')

# Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

