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
#define DDOS_RATE "100Mbps"   // Increased DDoS traffic rate
#define MAX_SIMULATION_TIME 100.0  // Increased simulation time for more packets

// Number of Bots
#define NUMBER_OF_BOTS 5  // Adjusted for balance

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("DDoSTrafficSimulation");

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

    // Nodes for attack bots
    NodeContainer botNodes;
    botNodes.Create(NUMBER_OF_BOTS);

    // Define the Point-To-Point Links and their Parameters
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
    p2p.SetChannelAttribute("Delay", StringValue("1ms"));

    // Install the Point-To-Point Connections between Nodes
    NetDeviceContainer routerServerLink, botDeviceContainer[NUMBER_OF_BOTS], iotDeviceRouterLink;
    routerServerLink = p2p.Install(router.Get(0), server.Get(0));
    iotDeviceRouterLink = p2p.Install(iotDevices.Get(0), router.Get(0));  // Legitimate device to router

    for (int i = 0; i < NUMBER_OF_BOTS; ++i)
    {
        botDeviceContainer[i] = p2p.Install(botNodes.Get(i), router.Get(0));
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

    Ipv4InterfaceContainer iotDeviceInterface = ipv4.Assign(iotDeviceRouterLink);
    Ipv4InterfaceContainer routerServerInterface = ipv4.Assign(routerServerLink);

    // DDoS Application Behaviour
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

    // Legitimate Traffic from IoT Device (TCP)
    BulkSendHelper bulkSend("ns3::TcpSocketFactory", InetSocketAddress(routerServerInterface.GetAddress(1), TCP_SINK_PORT));
    bulkSend.SetAttribute("MaxBytes", UintegerValue(0)); // Unlimited traffic
    bulkSend.SetAttribute("SendSize", UintegerValue(1024));
    ApplicationContainer bulkSendApp = bulkSend.Install(iotDevices.Get(0));
    bulkSendApp.Start(Seconds(0.0));
    bulkSendApp.Stop(Seconds(MAX_SIMULATION_TIME));

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
                                  "MinX", DoubleValue(0.0), "MinY", DoubleValue(0.0), "DeltaX", DoubleValue(5.0), "DeltaY", DoubleValue(10.0),
                                  "GridWidth", UintegerValue(5), "LayoutType", StringValue("RowFirst"));

    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");

    mobility.Install(router);
    mobility.Install(server);
    mobility.Install(botNodes);
    mobility.Install(iotDevices);

    // Create an AnimationInterface object for XML output
    AnimationInterface anim("DDoSTraffic.xml");

    // Enable packet metadata in XML
    anim.EnablePacketMetadata(true);

    // Positioning of nodes
    ns3::AnimationInterface::SetConstantPosition(router.Get(0), 15, 10);
    ns3::AnimationInterface::SetConstantPosition(server.Get(0), 25, 10);

    uint32_t x_pos = 0;
    for (int m = 0; m < NUMBER_OF_BOTS; ++m)
    {
        ns3::AnimationInterface::SetConstantPosition(botNodes.Get(m), x_pos++, 20);
    }
    ns3::AnimationInterface::SetConstantPosition(iotDevices.Get(0), 5, 5); // Position the legitimate IoT device

    // Run the Simulation
    Simulator::Run();
    Simulator::Destroy();
    return 0;
}
