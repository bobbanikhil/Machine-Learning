import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


df1 = pd.read_csv("season-2223_csv.csv")
df2 = pd.read_csv("season-2122_csv.csv")
df3 = pd.read_csv("season-2021_csv.csv")
df4 = pd.read_csv("season-1920_csv.csv")
df5 = pd.read_csv("season-1819_csv.csv")
df6 = pd.read_csv("season-1718_csv.csv")
df7 = pd.read_csv("season-1617_csv.csv")
df8 = pd.read_csv("season-1516_csv.csv")
df9 = pd.read_csv("season-1415_csv.csv")
df10 = pd.read_csv("season-1314_csv.csv")
df11 = pd.read_csv("season-1213_csv.csv")
df12 = pd.read_csv("season-1112_csv.csv")
df13 = pd.read_csv("season-1011_csv.csv")
df14 = pd.read_csv("season-0910_csv.csv")
df15 = pd.read_csv("season-0809_csv.csv")
df16 = pd.read_csv("season-0708_csv.csv")
df17 = pd.read_csv("season-0607_csv.csv")
df18 = pd.read_csv("season-0506_csv.csv")

df = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18])
df = df.reset_index(drop=True)
df.dropna(axis=1,inplace=True)

columns_to_drop = ['Div', 'Referee','Date','HTR','FTAG','FTHG','HTHG','HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
df = df.drop(columns = columns_to_drop)

label_encoder = LabelEncoder()
df['HomeTeam'] = label_encoder.fit_transform(df['HomeTeam'])
df['AwayTeam'] = label_encoder.transform(df['AwayTeam'])

X = df.drop('FTR', axis=1)
y = df['FTR']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model_rf = RandomForestClassifier(bootstrap=True, max_depth=10, max_features='sqrt', min_samples_leaf=4, min_samples_split=10, n_estimators=100, random_state=4)

# Train the model on the training data
model_rf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model_rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

with open('model_and_encoder.pkl', 'wb') as file:
    pickle.dump((model_rf, label_encoder), file)




