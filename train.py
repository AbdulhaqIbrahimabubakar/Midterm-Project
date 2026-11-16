# train.py (Trains & saves model.bin)
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

# Load & clean data
df = pd.read_csv('UCI_STUDENT_PERFORMANCE/student-mat.csv', sep=';')
df.columns = df.columns.str.lower().str.replace(' ', '_')
for c in df.select_dtypes(exclude='number').columns:
    df[c] = df[c].str.strip().str.lower().str.replace(' ', '_')

# Split 80/20
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
y_full_train = df_full_train.g3.values
y_test = df_test.g3.values

del df_full_train['g3']
del df_test['g3']

# Vectorize & train winning Gradient Boosting model
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(df_full_train.to_dict(orient='records'))
X_test = dv.transform(df_test.to_dict(orient='records'))

model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=2,
    random_state=1
)
model.fit(X_full_train, y_full_train)

preds = model.predict(X_test)
print(f"Test RMSE: {root_mean_squared_error(y_test, preds):.4f}")
print(f"Test R2:   {r2_score(y_test, preds):.4f}")

# Save vectorizer and model
with open('model.bin', 'wb') as f_out:
    pickle.dump((dv, model), f_out)
print("Saved model.bin successfully!")
