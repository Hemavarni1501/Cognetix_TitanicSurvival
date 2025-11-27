import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- 1. Load the Titanic Dataset ---
file_path = 'train.csv'
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Ensure it is in the same directory.")
    exit()

# Define features and target
TARGET = 'Survived'
FEATURES = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# --- 2. Handle Missing Values and Clean Data ---

# A. Missing Values
# Fix warnings by using direct assignment instead of inplace=True
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Cabin: Drop Cabin as it has too many missing values
df.drop('Cabin', axis=1, inplace=True)

# --- 3. Visualize Survival Rate by Group (EDA) ---
print("\n--- Generating Survival Rate Visualizations ---")

# A. Survival Rate by Gender
plt.figure(figsize=(6, 4))
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Rate')
plt.savefig('survival_by_gender.png')
# plt.show() # Uncomment to display plot locally

# B. Survival Rate by Passenger Class (Pclass)
plt.figure(figsize=(6, 4))
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Passenger Class')
plt.ylabel('Survival Rate')
plt.xlabel('Passenger Class (1st, 2nd, 3rd)')
plt.savefig('survival_by_class.png')
# plt.show()

# C. Survival Rate by Age Group
# Create age groups for visualization
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 18, 30, 50, 80], 
                       labels=['Child', 'Young Adult', 'Adult', 'Senior'])
plt.figure(figsize=(7, 5))
sns.barplot(x='AgeGroup', y='Survived', data=df)
plt.title('Survival Rate by Age Group')
plt.ylabel('Survival Rate')
plt.savefig('survival_by_agegroup.png')
# plt.show()

# --- 4. Feature Encoding for Modeling ---
# Convert categorical variables into dummy/indicator variables
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# --- 5. Select Final Features and Target for Model ---
# Drop the AgeGroup column and keep the encoded columns
FINAL_FEATURES = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 
                  'Sex_male', 'Embarked_Q', 'Embarked_S']

X = df[FINAL_FEATURES]
y = df[TARGET]

# --- 6. Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining set size: {len(X_train)}, Testing set size: {len(X_test)}")

# --- 7. Train a Classification Model (Random Forest) ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nModel trained successfully: Random Forest Classifier.")

# --- 8. Evaluate Model Performance ---
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- Model Evaluation Metrics ---")
print(f"Accuracy Score: {accuracy:.4f}")
print(f"Precision Score: {precision:.4f}")
print(f"Recall Score: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Optional: Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Died', 'Survived'], yticklabels=['Died', 'Survived'])
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix_titanic.png')
plt.show()