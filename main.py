import pandas as pd
import numpy as np
import json

# Load the dataset from the correct path
df = pd.read_csv('data/dataset.csv')

# Encode categorical variables
outlook_mapping = {'Sunny': 0, 'Overcast': 1, 'Rain': 2}
temperature_mapping = {'Hot': 0, 'Mild': 1, 'Cool': 2}
humidity_mapping = {'High': 0, 'Normal': 1}
wind_mapping = {'Weak': 0, 'Strong': 1}
play_tennis_mapping = {'No': 0, 'Yes': 1}

df['Outlook'] = df['Outlook'].map(outlook_mapping)
df['Temperature'] = df['Temperature'].map(temperature_mapping)
df['Humidity'] = df['Humidity'].map(humidity_mapping)
df['Wind'] = df['Wind'].map(wind_mapping)
df['PlayTennis'] = df['PlayTennis'].map(play_tennis_mapping)

# Helper function to calculate prior probabilities (P(class))
def calculate_prior_probabilities(df):
    total_instances = len(df)
    class_counts = df['PlayTennis'].value_counts()
    priors = class_counts / total_instances
    return priors

# Helper function to calculate likelihoods (P(feature | class)) with Laplace smoothing
def calculate_likelihoods(df, priors):
    likelihoods = {}
    # For each class (PlayTennis = 0 or 1)
    for class_value in priors.index:
        class_data = df[df['PlayTennis'] == class_value]
        class_likelihoods = {}
        
        # For each feature column (excluding the target column 'PlayTennis')
        for column in df.columns[:-1]:  # All columns except 'PlayTennis'
            column_values = df[column].unique()
            column_likelihoods = {}
            
            for value in column_values:
                # Count the occurrences of the feature value in this class
                feature_count = len(class_data[class_data[column] == value])
                total_class_count = len(class_data)
                # Apply Laplace smoothing (adding 1 to the count)
                smoothed_likelihood = (feature_count + 1) / (total_class_count + len(column_values))
                column_likelihoods[str(value)] = smoothed_likelihood  # Convert value to string
            
            class_likelihoods[column] = column_likelihoods
        
        likelihoods[str(class_value)] = class_likelihoods  # Convert class_value to string
    
    return likelihoods

# Calculate priors and likelihoods
priors = calculate_prior_probabilities(df)
likelihoods = calculate_likelihoods(df, priors)

# Prepare the model
model = {
    'priors': priors.to_dict(),
    'likelihoods': likelihoods
}

# Convert keys to strings for JSON serialization
model['priors'] = {str(k): v for k, v in model['priors'].items()}
model['likelihoods'] = {str(k): {str(inner_k): inner_v for inner_k, inner_v in v.items()} for k, v in model['likelihoods'].items()}

# Save the model to a JSON file
with open('naive_bayes_model.json', 'w') as f:
    json.dump(model, f)

print("Model trained and saved to 'naive_bayes_model.json'")

# Function to predict the class label for a new instance
def predict(instance, model):
    priors = model['priors']
    likelihoods = model['likelihoods']
    
    # Ensure that the instance keys are strings (they should match the feature names)
    instance = {str(k): v for k, v in instance.items()}
    
    # Calculate the posterior for each class
    posteriors = {}
    
    for class_value, prior in priors.items():
        posterior = np.log(prior)  # Start with log of prior
        
        # Multiply the likelihoods of each feature given the class
        for feature, value in instance.items():
            # Ensure that the feature name is a string and lookup the likelihood
            likelihood = likelihoods[str(class_value)].get(feature, {}).get(str(value), 0)
            posterior += np.log(likelihood) if likelihood > 0 else -np.inf  # Avoid log(0)
        
        posteriors[class_value] = posterior
    
    # Return the class with the highest posterior
    predicted_class = max(posteriors, key=posteriors.get)
    return predicted_class

# # Example test instance
# test_instance = {'Outlook': 0, 'Temperature': 2, 'Humidity': 1, 'Wind': 0}  # Example: Sunny, Cool, Normal, Weak
# model = json.load(open('naive_bayes_model.json', 'r'))  # Load the model
# predicted_class = predict(test_instance, model)
# print(f"Predicted Class: {predicted_class}")

# **Testing and Evaluation**: Implementing Leave-One-Out Cross-Validation (LOOCV)
def evaluate_model(df, model):
    true_labels = []
    predicted_labels = []
    confusion_matrix = np.zeros((2, 2), dtype=int)  # Initialize confusion matrix (2x2 for binary classification)
    
    # Leave-One-Out Cross-Validation
    for index, row in df.iterrows():
        # Create a test instance
        test_instance = row[:-1].to_dict()  # All features except 'PlayTennis'
        true_label = row['PlayTennis']
        
        # Predict using the model
        predicted_label = predict(test_instance, model)
        
        # Store the true and predicted labels
        true_labels.append(true_label)
        predicted_labels.append(int(predicted_label))
        
        # Update the confusion matrix
        true_class = int(true_label)
        predicted_class = int(predicted_label)
        confusion_matrix[true_class, predicted_class] += 1
    
    # Calculate accuracy manually
    correct_predictions = np.sum(np.array(true_labels) == np.array(predicted_labels))
    accuracy = correct_predictions / len(true_labels)
    
    return accuracy, confusion_matrix, true_labels, predicted_labels

# Evaluate the model
accuracy, conf_matrix, true_labels, predicted_labels = evaluate_model(df, model)

# Print the evaluation results
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)

# **Misclassification Analysis**: Analyze misclassified instances
misclassified = []
for i, (true, predicted) in enumerate(zip(true_labels, predicted_labels)):
    if true != predicted:
        misclassified.append({'Index': i, 'True Label': true, 'Predicted Label': predicted, 'Features': df.iloc[i, :-1].to_dict()})

print("\nMisclassified Instances:")
for item in misclassified:
    print(f"Index {item['Index']} - True Label: {item['True Label']}, Predicted Label: {item['Predicted Label']}")
    print(f"Features: {item['Features']}")