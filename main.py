import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("churn.csv")
print(df.head())
# See column names and data types
print("\n--- Dataset Info ---")
print(df.info())

# Check if any data is missing
print("\n--- Missing Values ---")
print(df.isnull().sum())

# See how many customers churned (1) or stayed (0)
print("\n--- Churn Distribution ---")
print(df['Churn'].value_counts())
# Step 1: Drop 'customerID' – not useful for prediction
df.drop('customerID', axis=1, inplace=True)

# Step 2: Convert 'TotalCharges' to numeric (some are blank)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Step 3: Drop rows with missing values
df.dropna(inplace=True)

# Step 4: Convert 'Churn' to numbers – Yes -> 1, No -> 0
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Step 5: Turn all other text (categorical) columns into numbers using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

print("\n--- Data After Cleaning ---")
print(df.head())

# Split features (X) and label (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n--- Data Split ---")
print(f"Training data: {X_train.shape}")
print(f"Testing data: {X_test.shape}")

# Step 6: Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
