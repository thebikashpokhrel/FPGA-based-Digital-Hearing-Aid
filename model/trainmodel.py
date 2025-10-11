from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

model = RandomForestClassifier(n_estimators=20, random_state=42)
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
print(f'Train Accuracy: {accuracy_score(y_train, y_train_pred):.4f}')
joblib.dump(model, 'audio_classifier_rf.pkl')