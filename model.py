import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Load the data from the file
df = pd.read_csv(r'C:\Users\karan\OneDrive\Desktop\jenkins\project\prediction_model\finalTrain.csv')

# Preprocessing
df = df.drop(['ID', 'Delivery_person_ID', 'Order_Date', 'Time_Orderd', 'Time_Order_picked', 'City', 'Vehicle_condition'], axis=1)
df.dropna(inplace=True)
df = pd.get_dummies(df, columns=['Weather_conditions', 'Road_traffic_density', 'Type_of_order', 'Type_of_vehicle', 'Festival'])
df['Delivery_person_Age'].fillna(df['Delivery_person_Age'].mean(), inplace=True)
df['Delivery_person_Ratings'].fillna(df['Delivery_person_Ratings'].mean(), inplace=True)
df['multiple_deliveries'].fillna(df['multiple_deliveries'].mean(), inplace=True)

X = df.drop('Time_taken (min)', axis=1)
y = df['Time_taken (min)']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
model = RandomForestRegressor()
model.fit(X_train_scaled, y_train)

# Serialize the trained model and scaler
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model, 'model.pkl')