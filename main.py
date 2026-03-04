# ==============================
#  A SPAM EMAIL CLASSIFIER
# Supervised Learning (Classification)
# ==============================

# -------- IMPORT LIBRARIES --------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -------- STEP 1: LOAD DATA --------
print("\nLoading Dataset...\n")

data = pd.read_csv("spam.csv")

print(data.head())


# -------- STEP 2: EXPLORATORY DATA ANALYSIS --------
print("\nDataset Info:\n")
print(data.info()) # Dispaly the dataset 

# Check for missing data 
print("\nMissing Values:\n")
print(data.isnull().sum())


# Count spam vs ham
plt.figure()
sns.countplot(x=data["label"])
plt.title("Spam vs Ham Distribution")
plt.show()


# -------- STEP 3: FEATURE ENGINEERING --------
# Add message length feature
data["length"] = data["message"].apply(len)

plt.figure()
sns.histplot(data["length"], bins=30)
plt.title("Message Length Distribution")
plt.show()


# -------- STEP 4: CONVERT LABELS --------
data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

print("\nConverted Labels:\n")
print(data.head())


# -------- STEP 5: TEXT TO NUMBERS --------
print("\nVectorizing Text Data...\n")

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(data["message"])
y = data["label"]


# -------- STEP 6: TRAIN TEST SPLIT --------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -------- STEP 7: TRAIN MODEL --------
print("\nTraining Model...\n")

model = MultinomialNB()
model.fit(X_train, y_train)


# -------- STEP 8: PREDICTION --------
predictions = model.predict(X_test)


# -------- STEP 9: EVALUATION --------
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))


# -------- CONFUSION MATRIX --------
cm = confusion_matrix(y_test, predictions)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ----- STEP 10: TEST CUSTOM EMAIL ----
print("\nTesting Custom Email...\n")

sample_email = [
    "Hello, meet me in the office by 10:00 AM"
]

sample_vector = vectorizer.transform(sample_email)

prediction = model.predict(sample_vector)

if prediction[0] == 1:
    print("Prediction: SPAM")
else:
    print("Prediction: NOT SPAM")