


import sys
import pdb
# sys.path.append("C:\\users\\mccar\\miniconda3\\lib\\site-packages")
# sys.path.append("c:\\users\\mccar\\appdata\\local\\programs\\python\\python311\\lib\\site-packages")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose as mlr
from sklearn.preprocessing import LabelEncoder



# Load data
# pd.set_option('display.max_columns', 60)
train_df = pd.read_csv("data/gtzan_music_genre/features_30_sec.csv")
train_df.head()
print(train_df.shape)
print(train_df.info())


# Preprocessing
train_df = train_df.drop(['filename', 'length'], axis = 1)
y = train_df['label']
X = train_df.drop('label', axis = 1)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# More preprocessing
label_encoder = LabelEncoder()
scale = MinMaxScaler()
scaled_data = scale.fit_transform(x_train)
x_train_sc = pd.DataFrame(scaled_data, columns = x_train.columns).values
scaled_data = scale.fit_transform(x_test)
x_test_sc = pd.DataFrame(scaled_data, columns = x_test.columns).values
y_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.fit_transform(y_test)


# One-hot encode target labels
num_classes = len(np.unique(y_train))  # Assuming there are 10 classes
 # One hot encode target values
y_train_one_hot = pd.get_dummies(y_train).values
y_test_one_hot = pd.get_dummies(y_test).values




# NN optimization SA
nn_model = mlr.NeuralNetwork(hidden_nodes=[16,10],
                                     activation='relu',
                                     algorithm='random_hill_climb',
                                     max_iters=100,
                                     bias=True,
                                     is_classifier=True,
                                     learning_rate=0.75,
                                     early_stopping=False,
                                     clip_max=5,
                                     max_attempts=100,
                                     random_state=42)

# Fit the model
nn_model.fit(x_train_sc, y_train_one_hot)

# Predict labels for train and test set
y_train_pred = nn_model.predict(x_train_sc)
y_test_pred = nn_model.predict(x_test_sc)

# Calculate accuracy
pdb.set_trace()
train_accuracy = accuracy_score(y_train_one_hot, y_train_pred)
test_accuracy = accuracy_score(y_test_one_hot, y_test_pred)
print(train_accuracy)
print(test_accuracy)