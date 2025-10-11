import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split

def extract_features(file_path, n_mfcc=13):
    audio, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

def load_dataset(data_dir):
    X = []
    y = []
    class_map = {'empty': 0, 'line_noise': 1, 'pure_voice': 2, 'noise_voice': 3}
    
    for cls in os.listdir(data_dir):
        cls_dir = os.path.join(data_dir, cls)
        for file in os.listdir(cls_dir):
            if file.endswith('.wav'):
                features = extract_features(os.path.join(cls_dir, file))
                X.append(features)
                y.append(class_map[cls])
    
    return np.array(X), np.array(y)

X, y = load_dataset('audio_dataset')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
print(f'Training samples: {len(X_train)}, Test samples: {len(X_test)}')