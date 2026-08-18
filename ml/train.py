from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Input features
X = iris.data

# Target labels
y = iris.target

print("Dataset loaded successfully!")
print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)