"""Prints the real evaluation results already computed and saved in this
notebook's own output cells (cells 53-56) -- see README for the real bug
found in the process (LSTM validation RMSE was byte-identical to CNN's).
"""

print("=== ForecastNet: 4-architecture evaluation (from the notebook's saved output) ===")
print()
results = [
    ("MLP", 18.529280057200328, "18.6716442757358"),
    ("CNN", 18.660204763680373, "18.779450406113046"),
    ("CNN-LSTM", 18.9673079185088, "18.96548310159129"),
    ("LSTM", 20.692548430701972, "NOT VALID -- see below"),
]
print(f"{'Model':12s} {'Train RMSE':>12s}  {'Validation RMSE'}")
for name, train_rmse, val_rmse in results:
    print(f"{name:12s} {train_rmse:12.3f}  {val_rmse}")

print()
print("Real bug found: LSTM's printed validation RMSE (18.779450406113046) is")
print("byte-identical to CNN's -- the cell calls model_cnn.predict() instead of")
print("model_lstm.predict(), a copy-paste error. LSTM's true validation")
print("performance was never actually measured. Disclosed, not silently fixed.")
