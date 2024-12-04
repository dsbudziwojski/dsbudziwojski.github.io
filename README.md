David Budziwojski

A&O Sci C111

Professor Lozinski

## Introduction
Traditional Computers use bits to represent data, which at a low level are acted upon by logic gates and are encodings that remain deterministic (either 0 or 1). However, by expanding the structure of computers and how they represent information, it is possible to leverage aspects of quantum mechanics to perform computations that leverage principles of superposition and entanglement. As a result, the bit is extended to a quantum bit (or qubit for short) that no longer remains deterministic throughout computations and serves as the fundamental means of representing information in a quantum computer. During a qubit's non-deterministic state, it remains in a superposition that collapses to 0 or 1 upon being measured. Moreover, the quantum phenomenon of entanglement plays a crucial role in computation, as it directly relates one qubit to another. While quantum computers are still a scarce resource, emulators of quantum computers exist, like the one provided by IBM in their Quiskit library. These emulators work by utilizing mathematical models to simulate quantum systems. As such there remain some flaws as quantum states are an inherently difficult thing to model; however, such resources still provide unique insights. Machine learning models utilize algorithms to decompose data to help obtain relationships and detect patterns that allow them to perform a variety of tasks. While the models utilized in this paper remain in the traditional realm of computing, a model can indirectly identify patterns that possibly lead to entanglement. 

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
Afterward, this dataset was loaded into the same file as our model. However, before use, the features and labels of the data set were seperated and then they were split into training and testing sets. Lastly, the data was processed one last time, where the features were normalized.
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

**Logisitc Regression**

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
   
The choice of using a neural network came down to the model’s adeptness in handling high-dimensional data and ability to identify complex relations that develop from such data. The TensorFlow library was used to implement a neural network with an input size corresponding to the number of features, an output size corresponding to the length of a single output, and a hidden layer of size 5. Mroeover, the compilation of the model utilized an adam optimization and a learning rate of 0.01. While a significant reason behind the use of a neural network was its complexity and its ability to model complex systems well, the choice of a hidden layer size 5 comes from the fact that the data set only contains 3 features. While an increase in size may allow the model to perform better, there is a concern about overfitting the data. As such, 5 was empirically selected after choosing various other sizes. 

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

The logistic regression's final preformance was an accuracy of 87.36% on the training data and 86.30% on the testing set. Below is a Cost vs. Epochs graph for the logistic regression as well as an ROC curve. 

![download](https://github.com/user-attachments/assets/206cfa0a-ae77-4843-afb6-3e9a5ca0963d)

![download](https://github.com/user-attachments/assets/321f8dc2-9c31-45ff-ac6f-ceecc3361699)

The neural network's final performance was an accuracy of 87.38% on the training data and 86.30% on the testing set. Below is a Cost vs. Epochs graph for the neural network.


![download](https://github.com/user-attachments/assets/fb964aa2-6f6a-4fa6-90f2-a6dda12ad1f1)

## Analysis

## Conclusion
