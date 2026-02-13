
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


X = np.load('X_data.npy')
y = np.load('y_data.npy')
classes = np.load('classes.npy')

#обучение 
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)


model_to_save = {
    'rf_model': model,
    'letter_names': classes
}
joblib.dump(model_to_save, 'models/kazakh_sign_model.pkl')
print("Готово! Модель и буквы упакованы вместе.")