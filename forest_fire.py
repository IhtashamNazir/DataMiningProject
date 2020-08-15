import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
from sklearn.externals import joblib
warnings.filterwarnings("ignore")

data = pd.read_csv("Forest_fire.csv")
data = np.array(data)

X = data[1:, 1:-1]
y = data[1:, -1]
y = y.astype('int')
X = X.astype('int')
# print(X,y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
model = LogisticRegression()


model.fit(X_train, y_train)

# inputt = [int(x) for x in "45 32 60".split(' ')]
# final = [np.array(inputt)]

# b = log_reg.predict_proba(final)
# print(b)
# print(final)
with open("pickle_model", "wb") as f:
    pickle.dump(model, f)
# joblib.dump(model, "joblib_model")

# pickle_out=open("dict.pickle","wb")

