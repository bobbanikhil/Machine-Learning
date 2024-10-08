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

columns_to_drop = ['Div', 'Referee','Date','HTR','FTAG','FTHG', 'HTR']
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

pickle.dump(model_rf, open('model_rf.pkl','wb'))

# Read the testing set
df_testing = pd.read_csv("Testing Set.csv")

# Drop NaN values and unnecessary columns (similar to what you did for the training data)
df_testing.dropna(axis=1, inplace=True)
 # Make sure to drop the same columns as in the training data

# Apply label encoding to categorical variables
df_testing['HomeTeam'] = label_encoder.transform(df_testing['HomeTeam'])
df_testing['AwayTeam'] = label_encoder.transform(df_testing['AwayTeam'])

# Split features and target variable
X_testing = df_testing.drop('FTR', axis=1)
y_testing = df_testing['FTR']

# Make predictions on the testing set using the trained model
y_pred_testing = model_rf.predict(X_testing)

# Evaluate the model on the testing set
accuracy_testing = accuracy_score(y_testing, y_pred_testing)
print("Accuracy on Testing Set:", accuracy_testing)

# Display classification report for the testing set
print("\nClassification Report for Testing Set:")
print(classification_report(y_testing, y_pred_testing))

# Display confusion matrix for the testing set
print("\nConfusion Matrix for Testing Set:")
print(confusion_matrix(y_testing, y_pred_testing))

# Print the predictions
print("Predictions on Testing Set:")
print(y_pred_testing)


