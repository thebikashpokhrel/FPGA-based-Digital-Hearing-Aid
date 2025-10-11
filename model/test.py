from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

model = joblib.load('audio_classifier_rf.pkl')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

y_pred = model.predict(X_test)
class_names = ['empty', 'line_noise', 'pure_voice', 'noise_voice']
print(f'Test Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(classification_report(y_test, y_pred, target_names=class_names))