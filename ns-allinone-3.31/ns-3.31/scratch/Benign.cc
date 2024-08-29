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

#define UDP_SINK_PORT 9001
#define TCP_SINK_PORT 9000

// Experimental parameters
#define TCP_RATE "10Gbps"
#define MAX_SIMULATION_TIME 1200.0
#define SEND_SIZE 64  // Further reduced packet size
#define NUM_TCP_CONNECTIONS 10  // Increase number of simultaneous TCP connections

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("BenignTrafficSimulation");

int main(int argc, char *argv[])
{
    CommandLine cmd;
    cmd.Parse(argc, argv);

    Time::SetResolution(Time::NS);
    LogComponentEnable("UdpEchoClientApplication", LOG_LEVEL_INFO);
    LogComponentEnable("UdpEchoServerApplication", LOG_LEVEL_INFO);

    // IoT Devices, Router, and Server
    NodeContainer iotDevices, router, server;
    iotDevices.Create(1);  // Create 1 legitimate IoT device
    router.Create(1);
    server.Create(1);

    // Define the Point-To-Point Links and their Parameters
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("10Gbps"));
    p2p.SetChannelAttribute("Delay", StringValue("0.1ms"));

    // Install the Point-To-Point Connections between Nodes
    NetDeviceContainer routerServerLink, iotDeviceRouterLink;
    routerServerLink = p2p.Install(router.Get(0), server.Get(0));
    iotDeviceRouterLink = p2p.Install(iotDevices.Get(0), router.Get(0));  // Legitimate device to router

    // Assign IP to IoT Devices and Server
    InternetStackHelper stack;
    stack.Install(router);
    stack.Install(server);
    stack.Install(iotDevices);

    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.1.0.0", "255.255.255.0");
    Ipv4InterfaceContainer iotDeviceInterface = ipv4.Assign(iotDeviceRouterLink);
    Ipv4InterfaceContainer routerServerInterface = ipv4.Assign(routerServerLink);

    // Create multiple TCP connections from IoT Device to Server
    ApplicationContainer bulkSendApps;
    for (int i = 0; i < NUM_TCP_CONNECTIONS; ++i) {
        BulkSendHelper bulkSend("ns3::TcpSocketFactory", InetSocketAddress(routerServerInterface.GetAddress(1), TCP_SINK_PORT + i));
        bulkSend.SetAttribute("MaxBytes", UintegerValue(0)); // Unlimited traffic
        bulkSend.SetAttribute("SendSize", UintegerValue(SEND_SIZE)); // Smaller packets to generate more
        bulkSendApps.Add(bulkSend.Install(iotDevices.Get(0)));
    }
    bulkSendApps.Start(Seconds(0.0));
    bulkSendApps.Stop(Seconds(MAX_SIMULATION_TIME));

    // TCP Sinks on receiver side (Server)
    ApplicationContainer TCPSinkApps;
    for (int i = 0; i < NUM_TCP_CONNECTIONS; ++i) {
        PacketSinkHelper TCPsink("ns3::TcpSocketFactory",
                                 Address(InetSocketAddress(Ipv4Address::GetAny(), TCP_SINK_PORT + i)));
        TCPSinkApps.Add(TCPsink.Install(server.Get(0)));
    }
    TCPSinkApps.Start(Seconds(0.0));
    TCPSinkApps.Stop(Seconds(MAX_SIMULATION_TIME));

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // Simulation NetAnim configuration and node placement
    MobilityHelper mobility;

    mobility.SetPositionAllocator("ns3::GridPositionAllocator",
                                  "MinX", DoubleValue(0.0), "MinY", DoubleValue(0.0), "DeltaX", DoubleValue(5.0), "DeltaY", DoubleValue(10.0),
                                  "GridWidth", UintegerValue(5), "LayoutType", StringValue("RowFirst"));

    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");

    mobility.Install(router);
    mobility.Install(server);
    mobility.Install(iotDevices);

    // Create an AnimationInterface object for XML output
    AnimationInterface anim("BenignTraffic.xml");

    // Enable packet metadata in XML
    anim.EnablePacketMetadata(true);

    // Positioning of nodes
    ns3::AnimationInterface::SetConstantPosition(router.Get(0), 15, 10);
    ns3::AnimationInterface::SetConstantPosition(server.Get(0), 25, 10);
    ns3::AnimationInterface::SetConstantPosition(iotDevices.Get(0), 5, 5); // Position the legitimate IoT device

    // Run the Simulation
    Simulator::Run();
    Simulator::Destroy();
    return 0;
}
