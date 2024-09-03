# FuzzAIoT 🚀
**AI-Driven Dynamic Fuzz Testing for IoT Security: Detection and Mitigation of DDoS Attacks Using Graph Neural Networks**

## Overview 🌐
`FuzzAIoT` is an advanced security framework designed to safeguard IoT networks against Distributed Denial of Service (DDoS) attacks. By leveraging Dynamic Fuzz Testing combined with Graph Neural Networks (GNNs), this project provides a cutting-edge solution for ensuring the integrity, availability, and reliability of IoT systems. The framework is built to detect and mitigate DDoS attacks in real-time, preserving network performance and protecting connected devices.

## Key Features ✨
- **Dynamic Fuzz Testing**: Simulates various attack scenarios to uncover vulnerabilities within IoT networks, providing a proactive approach to security.
- **Graph Neural Networks (GNNs)**: Utilizes GNNs trained on realistic network traffic to accurately detect and mitigate DDoS attacks with a high success rate.
- **NS3 Simulations**: Employs the NS3 network simulator to generate diversified and realistic network traffic, ensuring comprehensive training and validation of AI models.
- **Real-Time Mitigation**: Implements real-time blocking of malicious IPs identified during the simulation, ensuring uninterrupted and secure network operations.

## Project Structure 🗂️
```plaintext
FuzzAIoT/
├── data/                   # Datasets used for training and validation
├── models/                 # Trained GNN models
├── scripts/                # Scripts for data processing, model training, and evaluation
├── outputs/                # CSV files after converting NS3 simulation files
├── simulations/            # NS3 simulation scripts and configurations
├── results/                # Graphs and simulation results
├── README.md               # Project documentation
└── LICENSE                 # License file
```

## Installation & Setup 🛠️
1. **Clone the repository**:
   ```bash
   git clone https://github.com/footcricket05/FuzzAIoT.git
   cd FuzzAIoT
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run NS3 simulations** to generate the dataset:
   ```bash
   cd simulations/
   ./waf --run your_simulation_script
   ```

4. **Train & Evaluate the GNN model**:
   ```bash
   python scripts/GNN_Identification.py
   ```

## Usage 🖥️
- **Generate Network Traffic**: Use NS3 simulations to create a realistic IoT network environment, generating both benign and malicious traffic.
- **Train the Model**: Use the generated data to train the GNN model for accurate DDoS detection.
- **Deploy the Model**: Integrate the trained model into the NS3 simulation environment for real-time detection and mitigation of DDoS attacks.

## Results 📊
- **Detection Accuracy**: 74%
- **Mitigation Success**: 95%
- The model effectively differentiates between benign and malicious traffic while maintaining optimal network performance.

### **Screenshots** 📸
- **Training Graphs**: View the [training loss and accuracy graphs](./results/training_graphs.jpg) to understand the model’s learning process.
- **Simulation Visualization**: See the [simulation results](./results/simulation_visualization.jpg) to observe how the GNN model mitigates DDoS attacks in real-time.
- **Confusion Matrix**: Check out the [confusion matrix](./results/confusion_matrix.jpg) that illustrates the model’s performance in distinguishing between benign and malicious traffic.

> **Note**: To visualize these results, navigate to the `results/` directory where all output graphs and screenshots are stored.

## Contributing 🤝
We welcome contributions from the community! Feel free to fork this repository, create a feature branch, and submit a pull request. Your input is valuable in enhancing the robustness and effectiveness of `FuzzAIoT`.

## License 📄
This project is licensed under the `GNU Affero General Public License v3.0`. See the LICENSE file for more details.
