# Quantum Results from Classical Data 
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

