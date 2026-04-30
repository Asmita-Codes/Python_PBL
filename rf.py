import pickle
import pandas as pd

# 1. Load model and scaler
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

# with open("scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)

print("Model and Scaler loaded successfully")

# 2. Feature names (must EXACTLY match training)
feature_names = [
    "protocol_type", "service", "src_bytes", "land", "wrong_fragment",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "is_guest_login",
    "srv_serror_rate", "dst_host_srv_diff_host_rate",
    "dst_host_rerror_rate"
]

# 3. Show input format
print("\nEnter ALL values as NUMBERS only (no text).")
print(f"Total inputs required: {len(feature_names)}\n")

for i, name in enumerate(feature_names, start=1):
    print(f"{i}. {name}")

# 4. Take input
values = input("\nEnter values (space separated):\n")

# 5. Validate numeric input
try:
    numeric_features = [float(x) for x in values.strip().split()]
except ValueError:
    print("Error: Only numeric values are allowed.")
    exit()

# 6. Validate count
if len(numeric_features) != len(feature_names):
    print(f"Error: Expected {len(feature_names)} values, got {len(numeric_features)}")
    exit()

# 7. Convert to DataFrame
input_df = pd.DataFrame([numeric_features], columns=feature_names)

# 8. Scale input
try: 
    import numpy as np

    input_array = np.array(numeric_features).reshape(1,-1)
    # Scale ONLY values, then assign back to same DataFrame
    # input_df[feature_names] = scaler.transform(input_df[feature_names])
    # scaled_input = input_df
    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler()
    scaled_input=scaler.fit_transform(input_array)
    #scaled_input=pd.DataFrame(scaled_values,columns=feature_names)
except Exception as e:
    print(" Scaling error:", e)
    exit()

# 9. Predict
try:
    prediction = model.predict(scaled_input)
except Exception as e:
    print("Prediction error:", e)
    exit()

# 10. Probability (optional)
if hasattr(model, "predict_proba"):
    prob = model.predict_proba(scaled_input)

# 11. Output
print("\n--- RESULT ---")
print("Predicted Class:", prediction[0])

if hasattr(model, "predict_proba"):
    print("Class Probabilities:", prob)