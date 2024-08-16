# iris_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Load the dataset
df = pd.read_csv('iris.csv')

# Encode the species
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# Split the dataset
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model and encoder
with open('iris_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(le, le_file)

print("Model and encoder saved!")
