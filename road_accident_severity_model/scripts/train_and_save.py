# train_and_save.py
# Generates synthetic data, trains Linear Regression and saves the model.
import numpy as np, pandas as pd, pickle, os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def generate_synthetic(n=1000, seed=42):
    np.random.seed(seed)
    vehicle_speed = np.random.normal(60, 15, n).clip(5,160)
    weather = np.random.choice([0,1,2,3], size=n, p=[0.6,0.25,0.1,0.05])
    road = np.random.choice([0,1,2,3], size=n, p=[0.7,0.2,0.08,0.02])
    visibility = np.random.normal(1000,400,n).clip(50,5000)
    driver_age = np.random.normal(35,12,n).clip(16,90).astype(int)
    num_vehicles = np.random.poisson(1.2,n)+1
    time_of_day = np.random.choice([0,1,2,3], size=n, p=[0.25,0.35,0.2,0.2])
    alcohol = np.random.binomial(1,0.08,n)
    seatbelt = np.random.binomial(1,0.82,n)
    # severity generation (simplified)
    weather_effect = np.where(weather==0,0.0, np.where(weather==1,1.5, np.where(weather==2,3.0,4.0)))
    road_effect = np.where(road==0,0.0, np.where(road==1,1.8, np.where(road==2,2.5,4.2)))
    time_effect = np.where(time_of_day==3,1.2, np.where(time_of_day==2,0.8,0.2))
    age_effect = (driver_age>=65).astype(int)*0.7
    alcohol_effect = alcohol*3.8
    seatbelt_effect = -seatbelt*2.6
    speed_effect = 0.03*vehicle_speed
    visibility_effect = 0.15*(1000.0/visibility)
    vehicle_num_effect = (num_vehicles-1)*0.9
    base = 1.0
    severity = (base + speed_effect + weather_effect + road_effect + visibility_effect + time_effect + age_effect + alcohol_effect + seatbelt_effect + vehicle_num_effect + np.random.normal(scale=1.1,size=n))
    severity = np.clip(severity, 1.0, 10.0)
    df = pd.DataFrame({
        "vehicle_speed_kmph": np.round(vehicle_speed,2),
        "weather_condition": weather,
        "road_condition": road,
        "visibility_m": np.round(visibility,1),
        "driver_age": driver_age,
        "num_vehicles_involved": num_vehicles,
        "time_of_day": time_of_day,
        "alcohol_involved": alcohol,
        "seatbelt_used": seatbelt,
        "severity": np.round(severity,3)
    })
    return df

if __name__ == "__main__":
    out_root = os.path.join(os.path.dirname(__file__), "..")
    out_root = os.path.abspath(out_root)
    df = generate_synthetic(n=1000)
    os.makedirs(os.path.join(out_root,"data"), exist_ok=True)
    df.to_csv(os.path.join(out_root,"data","road_accident_data.csv"), index=False)
    # Preprocess and train
    df_enc = pd.get_dummies(df, columns=["weather_condition","road_condition","time_of_day"], drop_first=True)
    X = df_enc.drop(columns=["severity"])
    y = df_enc["severity"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    os.makedirs(os.path.join(out_root,"models"), exist_ok=True)
    payload = {"model": model, "features": X.columns.tolist()}
    with open(os.path.join(out_root,"models","accident_severity_model.pkl"), "wb") as f:
        pickle.dump(payload, f)
    print("Model trained and saved at", os.path.join(out_root,"models","accident_severity_model.pkl"))
