import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from ns3.core import Simulator
from ns3.network import NodeContainer
from ns3.internet import InternetStackHelper, Ipv4AddressHelper, Ipv4GlobalRoutingHelper
from ns3.point_to_point import PointToPointHelper
from ns3.applications import OnOffHelper, PacketSinkHelper, ApplicationContainer
from ns3.mobility import MobilityHelper
from ns3.netanim import AnimationInterface
import os

# Paths to your models
GNN_MODEL_PATH = "C:/Users/Shaurya/Downloads/FuzzAIoT/models/gnn_model.pth"
RL_MODEL_PATH = "C:/Users/Shaurya/Downloads/FuzzAIoT/models/rl_mitigation_model.pth"

# Load your trained models
class GNNModel(nn.Module):
    # Define your GNN model architecture
    def __init__(self):
        super(GNNModel, self).__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 2)  # Binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

gnn_model = GNNModel()
gnn_model.load_state_dict(torch.load(GNN_MODEL_PATH))
gnn_model.eval()

class RLModel(nn.Module):
    # Define your RL model architecture
    def __init__(self, state_size, action_size):
        super(RLModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

rl_model = RLModel(state_size=1, action_size=2)
rl_model.load_state_dict(torch.load(RL_MODEL_PATH))
rl_model.eval()

# Define the IoT network simulation
def run_simulation():
    # Set up the IoT network (using NS-3)
    iotDevices = NodeContainer()
    iotDevices.Create(5)

    routers = NodeContainer()
    routers.Create(1)

    server = NodeContainer()
    server.Create(1)

    # Set up Point-to-Point links
    p2p = PointToPointHelper()
    p2p.SetDeviceAttribute("DataRate", "100Mbps")
    p2p.SetChannelAttribute("Delay", "1ms")

    # Install devices and connections
    deviceContainer = [p2p.Install(iotDevices.Get(i), routers.Get(0)) for i in range(5)]
    serverLink = p2p.Install(routers.Get(0), server.Get(0))

    # Install internet stack
    internet = InternetStackHelper()
    internet.Install(iotDevices)
    internet.Install(routers)
    internet.Install(server)

    # Assign IP addresses
    ipv4 = Ipv4AddressHelper()
    ipv4.SetBase("10.0.0.0", "255.255.255.0")
    iotInterfaces = [ipv4.Assign(deviceContainer[i]) for i in range(5)]
    ipv4.Assign(serverLink)

    # Install applications
    onoff = OnOffHelper("ns3::UdpSocketFactory", routers.Get(0).GetObject(ns3.Ipv4.GetTypeId()).GetAddress(0, 0))
    onoff.SetAttribute("OnTime", ns3.StringValue("ns3::ConstantRandomVariable[Constant=1]"))
    onoff.SetAttribute("OffTime", ns3.StringValue("ns3::ConstantRandomVariable[Constant=0]"))
    onoff.SetConstantRate(ns3.DataRate("10Mbps"))
    apps = onoff.Install(iotDevices.Get(0))
    apps.Start(ns3.Seconds(1.0))
    apps.Stop(ns3.Seconds(10.0))

    # Start the simulation
    Simulator.Stop(ns3.Seconds(10.0))
    Simulator.Run()
    Simulator.Destroy()

    # Check for attack and apply mitigation
    with torch.no_grad():
        for i in range(5):
            packet_size = iotDevices.Get(i).GetObject(ns3.Ipv4.GetTypeId()).GetAddress(0, 0)
            packet_size = torch.tensor([float(packet_size)], dtype=torch.float32)
            output = gnn_model(packet_size)
            _, predicted = torch.max(output, 1)

            if predicted.item() == 1:  # If attack detected
                print(f"Attack detected on IoT device {i}. Applying mitigation...")
                state = torch.tensor([float(packet_size)], dtype=torch.float32)
                action_values = rl_model(state)
                action = torch.argmax(action_values).item()

                if action == 0:
                    print("Action: Block traffic.")
                    # Implement blocking logic here
                else:
                    print("Action: Throttle traffic.")
                    # Implement throttling logic here

if __name__ == "__main__":
    run_simulation()
