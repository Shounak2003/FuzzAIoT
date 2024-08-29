#include <ns3/csma-helper.h>
#include "ns3/mobility-module.h"
#include "ns3/nstime.h"
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/netanim-module.h"
#include <vector>
#include <string>

#define UDP_SINK_PORT 9001
#define TCP_SINK_PORT 9000

// Adjusted parameters
#define DDOS_RATE "512kb/s"  // Adjusted to balance with the increased benign traffic
#define TCP_RATE "10240kb/s" // Increased rate for benign traffic
#define MAX_SIMULATION_TIME 60.0  // Increased simulation time
#define NUMBER_OF_BOTS 5
#define NUMBER_OF_IOT_DEVICES 20  // Increased number of IoT devices for more benign traffic

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("IoTValidationScenarioFuzzTesting");

// Function to drop packets from malicious IPs
Ptr<NetDevice> GetDeviceFromIp(Ipv4Address ip, NodeContainer nodes)
{
    for (uint32_t i = 0; i < nodes.GetN(); ++i)
    {
        Ptr<Ipv4> ipv4 = nodes.Get(i)->GetObject<Ipv4>();
        int32_t ifIndex = ipv4->GetInterfaceForAddress(ip);
        if (ifIndex >= 0)
        {
            return ipv4->GetNetDevice(ifIndex);
        }
    }
    return nullptr;
}

int main(int argc, char *argv[])
{
    CommandLine cmd;
    cmd.Parse(argc, argv);

    Time::SetResolution(Time::NS);
    LogComponentEnable("UdpEchoClientApplication", LOG_LEVEL_INFO);
    LogComponentEnable("UdpEchoServerApplication", LOG_LEVEL_INFO);

    // Create IoT devices, routers, and servers
    NodeContainer iotDevices, router, server;
    iotDevices.Create(NUMBER_OF_IOT_DEVICES);  // Create multiple legitimate IoT devices
    router.Create(1);
    server.Create(1);

    // Nodes for attack bots
    NodeContainer botNodes;
    botNodes.Create(NUMBER_OF_BOTS);

    // Define the Point-To-Point Links and their Parameters
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
    p2p.SetChannelAttribute("Delay", StringValue("1ms"));

    // Install the Point-To-Point Connections between Nodes
    NetDeviceContainer routerServerLink, botDeviceContainer[NUMBER_OF_BOTS], iotDeviceRouterLinks[NUMBER_OF_IOT_DEVICES];
    routerServerLink = p2p.Install(router.Get(0), server.Get(0));

    for (int i = 0; i < NUMBER_OF_BOTS; ++i)
    {
        botDeviceContainer[i] = p2p.Install(botNodes.Get(i), router.Get(0));
    }

    // Connect each IoT device to the router
    for (uint32_t i = 0; i < NUMBER_OF_IOT_DEVICES; ++i)
    {
        iotDeviceRouterLinks[i] = p2p.Install(iotDevices.Get(i), router.Get(0));
    }

    // Assign IP to Bots and IoT Devices
    InternetStackHelper stack;
    stack.Install(router);
    stack.Install(server);
    stack.Install(botNodes);
    stack.Install(iotDevices);

    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.1.0.0", "255.255.255.0");  // Start a new subnet for bots
    Ipv4InterfaceContainer botInterfaces[NUMBER_OF_BOTS];
    for (int j = 0; j < NUMBER_OF_BOTS; ++j)
    {
        botInterfaces[j] = ipv4.Assign(botDeviceContainer[j]);
        ipv4.NewNetwork();  // Move to the next subnet for the next bot
    }

    Ipv4InterfaceContainer routerServerInterface = ipv4.Assign(routerServerLink);

    // Assign IP addresses to each IoT device
    for (uint32_t i = 0; i < NUMBER_OF_IOT_DEVICES; ++i)
    {
        ipv4.NewNetwork();
        Ipv4InterfaceContainer iotDeviceInterface = ipv4.Assign(iotDeviceRouterLinks[i]);
    }

    // List of malicious IPs detected by the GNN model
    std::vector<std::string> maliciousIps = {
        "10.1.0.1",  // Replace with actual IPs detected by your GNN model
        "10.1.1.2",
        "10.1.2.3"
        // Add more IPs as needed
    };

    // Applying the mitigation strategy
    for (const auto& ipStr : maliciousIps)
    {
        Ipv4Address ip(ipStr.c_str());
        Ptr<NetDevice> device = GetDeviceFromIp(ip, botNodes);
        if (device)
        {
            device->SetReceiveCallback(MakeNullCallback<bool, Ptr<NetDevice>, Ptr<const Packet>, uint16_t, const Address&>());
            NS_LOG_INFO("Dropped packets from IP: " << ipStr);
        }
    }

    // DDoS Application Behavior
    OnOffHelper onoff("ns3::UdpSocketFactory", Address(InetSocketAddress(routerServerInterface.GetAddress(1), UDP_SINK_PORT)));
    onoff.SetConstantRate(DataRate(DDOS_RATE));
    onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=30]"));
    onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
    ApplicationContainer onOffApp[NUMBER_OF_BOTS];

    for (int k = 0; k < NUMBER_OF_BOTS; ++k)
    {
        onOffApp[k] = onoff.Install(botNodes.Get(k));
        onOffApp[k].Start(Seconds(0.0));
        onOffApp[k].Stop(Seconds(MAX_SIMULATION_TIME));
    }

    // Legitimate Traffic from IoT Devices (TCP)
    for (uint32_t i = 0; i < NUMBER_OF_IOT_DEVICES; ++i)
    {
        BulkSendHelper bulkSend("ns3::TcpSocketFactory", InetSocketAddress(routerServerInterface.GetAddress(1), TCP_SINK_PORT));
        bulkSend.SetAttribute("MaxBytes", UintegerValue(0)); // Unlimited traffic
        bulkSend.SetAttribute("SendSize", UintegerValue(512)); // Smaller packets to generate more
        ApplicationContainer bulkSendApp = bulkSend.Install(iotDevices.Get(i));
        bulkSendApp.Start(Seconds(0.0));
        bulkSendApp.Stop(Seconds(MAX_SIMULATION_TIME));
    }

    // UDP Sink on receiver side (Server)
    PacketSinkHelper UDPsink("ns3::UdpSocketFactory",
                             Address(InetSocketAddress(Ipv4Address::GetAny(), UDP_SINK_PORT)));
    ApplicationContainer UDPSinkApp = UDPsink.Install(server.Get(0));
    UDPSinkApp.Start(Seconds(0.0));
    UDPSinkApp.Stop(Seconds(MAX_SIMULATION_TIME));

    // TCP Sink on receiver side (Server)
    PacketSinkHelper TCPsink("ns3::TcpSocketFactory",
                            Address(InetSocketAddress(Ipv4Address::GetAny(), TCP_SINK_PORT)));
    ApplicationContainer TCPSinkApp = TCPsink.Install(server.Get(0));
    TCPSinkApp.Start(Seconds(0.0));
    TCPSinkApp.Stop(Seconds(MAX_SIMULATION_TIME));

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // Simulation NetAnim configuration and node placement
    MobilityHelper mobility;

    mobility.SetPositionAllocator("ns3::GridPositionAllocator",
                                "MinX", DoubleValue(0.0), 
                                "MinY", DoubleValue(0.0), 
                                "DeltaX", DoubleValue(100.0),  // Horizontal spacing remains the same
                                "DeltaY", DoubleValue(400.0),  // Significantly increase vertical spacing
                                "GridWidth", UintegerValue(10), // Number of columns in the grid
                                "LayoutType", StringValue("RowFirst"));

    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");

    mobility.Install(router);
    mobility.Install(server);
    mobility.Install(botNodes);
    mobility.Install(iotDevices);

    // Create an AnimationInterface object for XML output
    AnimationInterface anim("DDoSim_AttackSim_Mitigation.xml");

    // Enable packet metadata in XML
    anim.EnablePacketMetadata(true);

    // Positioning of nodes
    uint32_t x_pos = 0;
    for (int m = 0; m < NUMBER_OF_BOTS; ++m)
    {
        ns3::AnimationInterface::SetConstantPosition(botNodes.Get(m), x_pos * 100.0, 800.0);  // Spread out bots more horizontally with significantly increased vertical spacing
        x_pos++;
    }

    x_pos = 0;
    for (uint32_t i = 0; i < NUMBER_OF_IOT_DEVICES; ++i)
    {
        ns3::AnimationInterface::SetConstantPosition(iotDevices.Get(i), x_pos * 100.0, 400.0);  // Spread out IoT devices more horizontally with significantly increased vertical spacing
        x_pos++;
    }

    ns3::AnimationInterface::SetConstantPosition(router.Get(0), 50.0, 1200.0); // Further increased vertical position for the router
    ns3::AnimationInterface::SetConstantPosition(server.Get(0), 950.0, 1200.0); // Further increased vertical position for the server

    // Run the Simulation
    Simulator::Run();
    Simulator::Destroy();
    return 0;
}
