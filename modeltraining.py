import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
import joblib

# ---------- Step 1: Load dataset ----------
df = pd.read_csv("combined_telangana_places.csv")  
# Columns: Place, Latitude, Longitude, Description, Category, Type

print("✅ Dataset loaded:", df.shape)

# ---------- Step 2: Haversine distance ----------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# ---------- Step 3: Generate features & target ----------
# Example: pick a sample start and destination to generate training target
start = (17.3850, 78.4867)    # Hyderabad
dest = (17.6663, 78.9443)     # Yadadri

df['Distance_Start'] = df.apply(lambda row: haversine(start[0], start[1], row['Latitude'], row['Longitude']), axis=1)
df['Distance_Dest'] = df.apply(lambda row: haversine(dest[0], dest[1], row['Latitude'], row['Longitude']), axis=1)

# Target score: closer to the line → higher score
df['Target_Score'] = 1 / (df['Distance_Start'] + df['Distance_Dest'] + 1)

# ---------- Step 4: Encode categorical features ----------
le = LabelEncoder()
df['Category_Encoded'] = le.fit_transform(df['Category'])

X = df[['Distance_Start', 'Distance_Dest', 'Category_Encoded']]
y = df['Target_Score']

# ---------- Step 5: Train-test split ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- Step 6: Train Random Forest ----------
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ---------- Step 7: Evaluate ----------
y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"R2: {r2_score(y_test, y_pred):.4f}")

# ---------- Step 8: Save Model ----------
joblib.dump(model, "combined_places_model.pkl")
joblib.dump(le, "category_encoder.pkl")
print("💾 Model and encoder saved successfully!")
