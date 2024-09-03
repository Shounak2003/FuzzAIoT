# FuzzAIoT 🚀
**AI-Driven Dynamic Fuzz Testing for IoT Security: Detection and Mitigation of DDoS Attacks Using Graph Neural Networks**

## Overview 🌐
`FuzzAIoT` is an advanced security framework designed to detect and mitigate Distributed Denial of Service (DDoS) attacks in IoT networks. By leveraging Dynamic Fuzz Testing combined with Graph Neural Networks (GNNs), this project provides a robust defense mechanism to ensure the integrity, availability, and reliability of IoT systems.

## Features ✨
- **Dynamic Fuzz Testing**: Uncovers potential vulnerabilities in IoT networks by simulating various attack scenarios.
- **Graph Neural Networks**: Trained to detect and mitigate DDoS attacks in real-time with high accuracy.
- **NS3 Simulations**: Generates realistic network traffic for training and validating the AI models.
- **Real-Time Mitigation**: Automatically blocks malicious IPs detected during simulation, ensuring network performance is not compromised.

## Project Structure 🗂️
```plaintext
FuzzAIoT/
├── data/                   # Datasets used for training and validation
├── models/                 # Trained GNN models
├── scripts/                # Scripts for data processing, model training, and evaluation
├── outputs/                # CSV files after converting NS3 simulation files
├── README.md               # Project documentation
└── LICENSE                 # License file
```

## Installation & Setup 🛠️
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/FuzzAIoT.git
   cd FuzzAIoT
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run NS3 simulations** to generate dataset:
   ```bash
   cd simulations/
   ./waf --run your_simulation_script
   ```

4. **Train the GNN model**:
   ```bash
   python scripts/train_gnn.py
   ```

5. **Evaluate the model**:
   ```bash
   python scripts/evaluate_gnn.py
   ```

## Usage 🖥️
- **Generate Network Traffic**: Use NS3 simulations to create a realistic IoT network environment and generate both benign and malicious traffic.
- **Train the Model**: Use the generated data to train the GNN model for accurate DDoS detection.
- **Deploy the Model**: Integrate the trained model into the NS3 simulation environment for real-time detection and mitigation of DDoS attacks.

## Results 📊
- **Detection Accuracy**: 74%
- **Mitigation Success**: 95%
- The model effectively differentiates between benign and malicious traffic while maintaining optimal network performance.

## Contributing 🤝
Contributions are welcome! Please fork this repository, create a feature branch, and submit a pull request.

## License 📄
This project is licensed under the `GNU Affero General Public License v3.0`. See the LICENSE file for details.

