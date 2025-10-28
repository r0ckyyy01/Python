# predict_example.py
# Load the saved model and run a sample prediction
import pickle, numpy as np, os
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "accident_severity_model.pkl")
with open(model_path, "rb") as f:
    payload = pickle.load(f)
model = payload["model"]
features = payload["features"]
# Example input (adjust as needed)
hypo = {
    "vehicle_speed_kmph": 85.0,
    "visibility_m": 600.0,
    "driver_age": 40,
    "num_vehicles_involved": 2,
    "alcohol_involved": 1,
    "seatbelt_used": 0,
    # one-hot fields left as zero by default; set e.g. 'weather_condition_2':1 for Fog
}
x = [0.0]*len(features)
feat_index = {f:i for i,f in enumerate(features)}
for k,v in hypo.items():
    if k in feat_index:
        x[feat_index[k]] = v
# set weather_condition_2 (Fog) and road_condition_1 (Wet) if present
if 'weather_condition_2' in feat_index:
    x[feat_index['weather_condition_2']] = 1.0
if 'road_condition_1' in feat_index:
    x[feat_index['road_condition_1']] = 1.0
pred = model.predict([x])[0]
print("Predicted severity (raw):", pred)
print("Predicted severity (clipped 1-10):", max(1.0, min(10.0, pred)))
