David Budziwojski

A&O Sci C111

Professor Lozinski

## Introduction
Traditional Computers use bits to represent data, which at a low level are acted upon by logic gates and are encodings that remain deterministic (either 0 or 1). However, by expanding the structure of computers and how they represent information, it is possible to leverage aspects of quantum mechanics to perform computations that utilize principles of superposition and entanglement. As a result, the bit is extended to a quantum bit (or qubit for short) that no longer remains deterministic throughout computations and serves as the fundamental means of representing information in a quantum computer. During a qubit's non-deterministic state, it remains in a superposition that collapses to 0 or 1 upon being measured. Moreover, the quantum phenomenon of entanglement plays a crucial role in computation, as it directly relates one qubit to another[1]. While quantum computers are still a scarce resource, emulators of quantum computers exist, like the one provided by IBM in their Quiskit library. These emulators work by utilizing mathematical models to simulate quantum systems. As such there remain some flaws as quantum states are an inherently difficult thing to model; however, such resources still provide unique insights. Machine learning models utilize algorithms to decompose data to help obtain relationships and detect patterns that allow them to perform a variety of tasks. While the models utilized in this article remain in the traditional realm of computing, a model can indirectly identify patterns that possibly lead to entanglement. 

## Data
A quantum gate emulator was used to synthetically create data that was later used to train a logistic regression and a neural network. The data began in a “classical” form (contained in a 3 x 5000 array) and was a value between 0 and 1. This value was then encoded into a quantum circuit through the use of angle encoding (essentially turning our data into “quantum form”) and was measured (collapsing the circuit) into a binary string of length 3. This string was then converted into integer form using binary to integer conversion and saved as a measurement of the given row. This process enabled the creation of the data set used: quantum_generated_data.csv. The data set consists of 3 features, which correspond to classical values between 0 and 1 that eventually collapse to an integer after being measured. 

`````
# Parameters
num_samples = 5000
num_features = 3
random_state = 42

np.random.seed(random_state)
classical_data = np.random.rand(num_samples, num_features)

def encode_data_into_circuit(data_point):
    """
    Encodes a single data point into a quantum circuit
    using angle encoding.
    """
    qc = QuantumCircuit(num_features)
    for i, feature in enumerate(data_point):
        qc.ry(feature * np.pi, i)
    qc.measure_all()
    return qc

quantum_circuits = [encode_data_into_circuit(data) for data in classical_data]

simulator = AerSimulator()
measurements = []

for qc in quantum_circuits:
    # Simulate circuit and get result
    result = simulator.run(qc, shots=1).result()
    counts = result.get_counts(qc)
    measurement = int(list(counts.keys())[0], 2)
    measurements.append(measurement)

quantum_dataset = pd.DataFrame(classical_data, columns=[f"Feature_{i+1}" for i in range(num_features)])
quantum_dataset['Measurement'] = measurements

quantum_dataset.to_csv('quantum_generated_data.csv', index=False)
`````
Afterward, this dataset was loaded into the same file as our model. However, before use, the features and labels of the data set were separated and then they were split into training and testing sets. Lastly, the data was processed one last time, where the features were normalized.
`````
dataset_path = 'quantum_generated_data.csv'
data = pd.read_csv(dataset_path)

# Separate the features (X) and the labels (y)
X = data.iloc[:, :-3].values  # Assuming the last 3 columns are labels
y = data.iloc[:, -3:].idxmax(axis=1).values  # Convert one-hot encoded labels to single-class labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
`````

## Model
Two types of machine learning models were used (1) a ***logistic regression*** and (2) a ***neural network***.

**Logistic Regression**

While a somewhat simple model, Logistic Regression was selected as a baseline model for future comparison. Not only is the model quick to train, but it does especially well with linear relationships; which could also be its drawback depending on the data. However, since the data set had only 3 features, the model was assumed to still do quite well despite the possibility of the data not being fully linear. The Scikit-learn library was used to implement the logistic regression, specifically, it was a multinomial logistic regression that utilized Limited-memory BFGS as its cost function. 

`````
# Initialize the Logistic Regression model
lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1, warm_start=True, random_state=42)

# Train the model and track cost (log-loss) over epochs
epochs = 500
training_costs = []
for epoch in range(epochs):
    lr.fit(X_train, y_train)
    y_train_pred_proba = lr.predict_proba(X_train)
    cost = log_loss(y_train, y_train_pred_proba)
    training_costs.append(cost)

# Make predictions on the test set
y_test_pred = lr.predict(X_test)
y_test_pred_proba = lr.predict_proba(X_test)

# Evaluate the model
train_accuracy = accuracy_score(y_train, lr.predict(X_train))
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\nFinal Performance:")
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
`````

**Neural Network**
   
The choice of using a neural network came down to the model’s adeptness in handling high-dimensional data and ability to identify complex relations that develop from such data. The TensorFlow library was used to implement a neural network with an input size corresponding to the number of features, an output size corresponding to the length of a single output, and a hidden layer of size 5. Moreover, the compilation of the model utilized an adam optimization and a learning rate of 0.01. While a significant reason behind the use of a neural network was its complexity and its ability to model complex systems well, the choice of a hidden layer size 5 comes from the fact that the data set only contains 3 features. While an increase in size may allow the model to perform better, there is a concern about overfitting the data. As such, 5 was empirically selected after running the model with various other sizes. 

`````
# Initialize and define the neural network
print("Initializing TensorFlow neural network...")
input_size = X_train.shape[1]
output_size = y.shape[1]
hidden_size = 5

model = Sequential([
    Dense(hidden_size, activation='sigmoid', input_shape=(input_size,)),
    Dense(output_size, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
print("Training...")
history = model.fit(X_train, y_train, epochs=500, batch_size=32, verbose=1)
print("Training finished")
`````

## Results

The logistic regression's final performance was an accuracy of 87.36% on the training data and 86.30% on the testing set. Looking at the Cost vs. Epochs graph for the logistic regression, the curve flatten outs around 0.4 quite rapidly. 

<div align="center">
  <img src="https://github.com/user-attachments/assets/206cfa0a-ae77-4843-afb6-3e9a5ca0963d" alt="download" width="600" height="500"/>
</div>

Created using:

```
# Plot Cost vs Epochs
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), training_costs, label="Training Cost (Log-Loss)")
plt.title("Cost vs Epochs (Logistic Regression)")
plt.xlabel("Epochs")
plt.ylabel("Log-Loss")
plt.legend()
plt.show()
```

The ROC curve for the logistic regression below labels three classes: 0, 1, and 2. These AUC values of a class correspond to how well the model distinguishes samples from the given class in comparison to those from the other two classes. As noted in the legend of the graph: Class 0 has an AUC of 0.75, Class 1 has an AUC of 0.68, and Class 2 has an AUC of 0.73. 

<div align="center">
  <img src="https://github.com/user-attachments/assets/321f8dc2-9c31-45ff-ac6f-ceecc3361699" alt="download" width="600" height="500"/>
</div>

Created using:

```
# Binarize y_test for ROC curve calculations
y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))

# Plot the ROC curve for each class
plt.figure(figsize=(10, 8))
for i in range(y_test_pred_proba.shape[1]):
    if np.sum(y_test_binarized[:, i]) == 0:
        print(f"Skipping class {i} (no positive samples in y_test)")
        continue
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_test_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

# Plot the random guessing line
plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing")

# Customize the plot
plt.title("ROC Curve (Logistic Regression)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()
```

The neural network's final performance was an accuracy of 87.38% on the training data and 86.30% on the testing set. Looking at the Cost vs. Epochs graph for the neural network, the curve flattens slowly towards 3.2 (and possibly lower given more epochs).

<div align="center">
  <img src="https://github.com/user-attachments/assets/da135b52-1389-48b1-9d13-7ed71ac7f487" alt="download" width="600" height="500"/>
</div>

Created Using:

```
# Plot the cost (loss) function over epochs
plt.plot(history.history['loss'])
plt.title('Cost vs Epochs (Neural Network)')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.show()
```

Similar to the logistic regression, the ROC curve below labels the same three classes, and as noted in the legend of the graph: Class 0 has an AUC of 0.74, Class 1 has an AUC of 0.68, and Class 2 has an AUC of 0.73. 

<div align="center">
  <img src="https://github.com/user-attachments/assets/99749f86-53ae-430a-b3e4-a8d048001d43" alt="download" width="600" height="500"/>
</div>

Created using:
```
# Get predicted probabilities for the test set
y_pred = model.predict(X_test)

# Convert continuous predictions to class labels for ROC curve calculation
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Plot the ROC curve for each class
plt.figure(figsize=(10, 8))
for i in range(y_test.shape[1]):
    if np.sum(y_test_labels == i) == 0:
        print(f"Skipping class {i} (no positive samples in y_test)")
        continue

    # Calculate FPR, TPR, and AUC for each class
    fpr, tpr, _ = roc_curve(y_test_labels, y_pred[:, i], pos_label=i)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

# Add the random guessing line
plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing")

# Customize the plot
plt.title("ROC Curve (Neural Network)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()
plt.show()
```

## Analysis

In terms of the accuracy of both models, they had identical accuracies of 86.30 on the testing result and a slight difference of .02% on the training data. This hints at the notion that both models do a fairly similar job in their respective predictions. Nonetheless, in both cases, the accuracies are moderate and are aspects that need to be further refined by either changing models or adding complexity to the current models. The Cost vs. Epochs graph, a visualization for the progression of costs of a model over multiple passes through the data, for the logistic regression illustrated a rapid decrease and early convergence of the cost function. While the neural network's Cost vs. Epochs graph also had its cost function decrease, however, the rate was substantially slower in comparison despite 250 more epochs. There are a couple of possibilities for the slower decrease of the cost function such as issues with the learning rate being too small and underfiting which begins to hint the current arrangment of the model not be "calibrated" quite well enough for the data set used. Looking at the ROC curves for both, they also look fairly similar to one another. ROC curves are a means of measuring the performance of a given model, comparing True Positive Rates and False Positive Rates. The AUC values for cases 1 and 2 are the same for both at 0.68 and 0.73 respectively. Case 0 is 0.01 higher for the logistic regression at 0.75. Area Under the Curve, AUC, is a means of determining the probability that a model will "outrank" positive classes in comparison to negative ones. While in both cases, the respective values indicate better performance than guessing randomly, the results are moderate and leave room for improvement (for both models) as both are still closer to randomness rather than unity. 

## Conclusion

The similar results by both models hint at the idea the data is possibly linearly related in some manner because logistic regression performs rather well on such data, as was seen in this case. As a result, this brings to question whether the use of such a complex model like a neural network is justifiable in this case, especially since a much simpler model performed comparingly well. In some sense, it is not justifiable in this case where there are only 3 features, rather the use of logistic regression may have been the better choice. However, if more features were added (and hence increasing the complexity of the relationship between features and measurement), this decision may not be suitable; especially, if the relationships are not as linear (which is highly likely). Moreover, it is apparent that there was room for improvement in terms of adjusting the hyperparameters of the neural network since for the most part they were empirically adjusted (trying to increase the accuracy without drastically increasing complexity). Some possible changes that could be added would be the addition of features in the data set and the usage of different model types. A point of concern to think about in terms of the data is that it was created using (initially) a random number generator which is not truly random and may have led to unforeseen patterns arising within the synthetic data set. Ultimately, the logistic regression and the neural network perform decently at relating classical data to pseudo-quantum data. Nonetheless, there is an underlying idea illustrated after the results: a complex model does not necessarily mean that it is better. The quantum realm is still quite unexplored and while some aspects of qubits were discussed, there is still significant to learn before quantum computers become more common.

## References

1. Bernhardt, C. (2019). Quantum Computing for Everyone. MIT Press. ISBN: 978-0262039253.

