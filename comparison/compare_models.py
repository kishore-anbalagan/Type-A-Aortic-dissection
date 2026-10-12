print("\n----------------------------------------------")
print("3D U-Net            0.2001         0.1111")
print("V-Net               0.4618         0.3007")
print("3D SegResNet        0.6111         0.4408")
print("----------------------------------------------")

results = {
    "3D U-Net": 0.2001,
    "V-Net": 0.4618,
    "3D SegResNet": 0.6111
}

best_model = max(results, key=results.get)

print("\nCurrent Best Model:")
print(best_model)
print(f"Dice Score: {results[best_model]:.4f}")

print("\n==============================================")
print("             MODEL COMPARISON")
print("==============================================")

print("Best model based on Dice Score:", best_model)
print("==============================================")