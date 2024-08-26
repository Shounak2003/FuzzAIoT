#include "ns3/mobility-module.h"
#include "ns3/nstime.h"
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/netanim-module.h"

#define TCP_SINK_PORT 9000
#define UDP_SINK_PORT 9001

// Parameters to change
#define BULK_SEND_MAX_BYTES 2097152
#define ATTACKER_DoS_RATE "20480kb/s"
#define MAX_SIMULATION_TIME 30.0

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("IoTSecurityLowRateDoS");

int main(int argc, char *argv[])
{
    CommandLine cmd;
    cmd.Parse(argc, argv);

    Time::SetResolution(Time::NS);
    LogComponentEnable("UdpEchoClientApplication", LOG_LEVEL_INFO);
    LogComponentEnable("UdpEchoServerApplication", LOG_LEVEL_INFO);

    // Create IoT devices, routers, and servers
    NodeContainer iotDevices;
    iotDevices.Create(4); // Adjust this number based on your IoT network setup

    NodeContainer routers;
    routers.Create(2);

    NodeContainer server;
    server.Create(1);

    // Define the Point-To-Point Links and their Parameters
    PointToPointHelper pp1, pp2;
    pp1.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
    pp1.SetChannelAttribute("Delay", StringValue("1ms"));

    pp2.SetDeviceAttribute("DataRate", StringValue("1.5Mbps"));
    pp2.SetChannelAttribute("Delay", StringValue("20ms"));

    // Install the Point-To-Point Connections between Nodes
    NetDeviceContainer d01, d12, d23;
    d01 = pp1.Install(iotDevices.Get(0), routers.Get(0));  // Legitimate IoT device to router
    d12 = pp1.Install(routers.Get(0), routers.Get(1));     // Router to router
    d23 = pp2.Install(routers.Get(1), server.Get(0));      // Router to server

    // Internet stack installation
    InternetStackHelper stack;
    stack.Install(iotDevices);
    stack.Install(routers);
    stack.Install(server);

    // IP Address Assignment
    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer i01 = ipv4.Assign(d01);

    ipv4.SetBase("10.1.2.0", "255.255.255.0");
    Ipv4InterfaceContainer i12 = ipv4.Assign(d12);

    ipv4.SetBase("10.1.3.0", "255.255.255.0");
    Ipv4InterfaceContainer i23 = ipv4.Assign(d23);

    // Attacker Application
    OnOffHelper onoff("ns3::UdpSocketFactory", Address(InetSocketAddress(i23.GetAddress(1), UDP_SINK_PORT)));
    onoff.SetConstantRate(DataRate(ATTACKER_DoS_RATE));
    onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=30]"));
    onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
    ApplicationContainer onOffApp = onoff.Install(iotDevices.Get(1)); // Assume second IoT device is compromised
    onOffApp.Start(Seconds(0.0));
    onOffApp.Stop(Seconds(MAX_SIMULATION_TIME));

    // Legitimate Sender Application
    BulkSendHelper bulkSend("ns3::TcpSocketFactory", InetSocketAddress(i23.GetAddress(1), TCP_SINK_PORT));
    bulkSend.SetAttribute("MaxBytes", UintegerValue(BULK_SEND_MAX_BYTES));
    ApplicationContainer bulkSendApp = bulkSend.Install(iotDevices.Get(0)); // Assume first IoT device is legitimate
    bulkSendApp.Start(Seconds(0.0));
    bulkSendApp.Stop(Seconds(MAX_SIMULATION_TIME - 10));

    // UDP Sink on the receiver side
    PacketSinkHelper UDPsink("ns3::UdpSocketFactory", Address(InetSocketAddress(Ipv4Address::GetAny(), UDP_SINK_PORT)));
    ApplicationContainer UDPSinkApp = UDPsink.Install(server.Get(0));
    UDPSinkApp.Start(Seconds(0.0));
    UDPSinkApp.Stop(Seconds(MAX_SIMULATION_TIME));

    // TCP Sink Application on the server side
    PacketSinkHelper TCPsink("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), TCP_SINK_PORT));
    ApplicationContainer TCPSinkApp = TCPsink.Install(server.Get(0));
    TCPSinkApp.Start(Seconds(0.0));
    TCPSinkApp.Stop(Seconds(MAX_SIMULATION_TIME));

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // Simulation NetAnim
    MobilityHelper mobility;
    mobility.SetPositionAllocator("ns3::GridPositionAllocator",
                                  "MinX", DoubleValue(0.0), "MinY", DoubleValue(0.0), "DeltaX", DoubleValue(5.0), "DeltaY", DoubleValue(10.0),
                                  "GridWidth", UintegerValue(3), "LayoutType", StringValue("RowFirst"));

    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(iotDevices);
    mobility.Install(routers);
    mobility.Install(server);

    AnimationInterface anim("IoTSecurityLowRateTCPDoS.xml");

    // Enable packet metadata in XML
    anim.EnablePacketMetadata(true);

    // Position nodes in the simulation
    ns3::AnimationInterface::SetConstantPosition(iotDevices.Get(0), 0, 0);
    ns3::AnimationInterface::SetConstantPosition(iotDevices.Get(1), 10, 0);
    ns3::AnimationInterface::SetConstantPosition(routers.Get(0), 5, 10);
    ns3::AnimationInterface::SetConstantPosition(routers.Get(1), 15, 10);
    ns3::AnimationInterface::SetConstantPosition(server.Get(0), 20, 10);

    // Run the Simulation
    Simulator::Run();
    Simulator::Destroy();
    return 0;
}
