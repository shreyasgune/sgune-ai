import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

#FAKE the dataset
X = np.array(
    [
        [180, 0.9, 0.5],   # Rock
        [175, 0.8, 0.4],
        [120, 0.7, 0.9],   # Pop
        [115, 0.6, 0.85],
        [90, 0.4, 0.3],    # Jazz
        [85, 0.3, 0.4]
    ]
)

y = np.array([0, 0, 1, 1, 2, 2]) # 0=rock, 1=pop, 2=jazz
genre = ["rock", "pop", "jazz"]

#Lets Train our model with fake data
clf = RandomForestClassifier(
    n_estimators=50,
    random_state=42
)
clf.fit(X,y)

# Lets save the model so our API doesnt re train every time we run it
joblib.dump(
    {
        "model": clf,
        "genre": genre,
    },
    "music_model.pkl"
)
print("music genre model trained and saved!")

