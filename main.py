import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Data loading and cleaning
def load_and_clean_data():
    # Load the cleaned dataset
    data = pd.read_csv("cleaned.csv")
    
    # Drop rows with null values and duplicates
    data_cleaned = data.dropna().drop_duplicates()
    
    print("Dataset Head:")
    print(data_cleaned.head())
    
    # Basic information about the dataset
    print("\nDataset Info:")
    print(data_cleaned.info())
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(data_cleaned.describe())
    
    return data_cleaned

# Feature extraction and model training
def train_model(data):
    # Separate features (X) and target variable (y)
    X = data.drop(columns=['Level'])
    y = data['Level']
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Initialize the Random Forest Classifier
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

# Feature importance plot
def plot_feature_importance(model, data):
    # Feature importance
    feature_importance = model.feature_importances_
    print("\nFeature Importance:")
    for col, importance in zip(data.columns[:-1], feature_importance):
        print(f"{col}: {importance:.4f}")
    
    # Plotting the feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance, y=data.columns[:-1])
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.show()

# Plot the count of interaction levels
def plot_interaction_levels(data):
    plt.figure(figsize=(8, 6))
    sns.countplot(x="Level", data=data, palette="viridis")
    plt.title("Distribution of Interaction Levels")
    plt.xlabel("Interaction Level")
    plt.ylabel("Count")
    plt.show()

if __name__ == "__main__":
    # Load and clean the data
    data = load_and_clean_data()
    
    # Train the model
    model = train_model(data)
    
    # Plot feature importance
    plot_feature_importance(model, data)
    
    # Plot interaction levels
    plot_interaction_levels(data)
