import xml.etree.ElementTree as ET
import csv
import re

def parse_xml_to_csv(xml_file, csv_file, label):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Source", "Destination", "PacketSize", "Label"])

        packet_count = 0  # To count the number of packets processed

        for packet in root.findall('.//p'):
            meta_info = packet.get('meta-info')
            if meta_info:
                # Extracting timestamp
                timestamp = packet.get('fbTx', '0')  # fallback to '0' if 'fbTx' not found
                
                # Extracting source and destination IP addresses
                ip_match = re.search(r'(\d+\.\d+\.\d+\.\d+) > (\d+\.\d+\.\d+\.\d+)', meta_info)
                if ip_match:
                    source_ip = ip_match.group(1)
                    destination_ip = ip_match.group(2)
                else:
                    continue  # skip this packet if IP addresses are not found

                # Extracting packet size
                size_match = re.search(r'length:\s+(\d+)', meta_info)
                packet_size = size_match.group(1) if size_match else '0'

                # Writing to CSV
                writer.writerow([timestamp, source_ip, destination_ip, packet_size, label])
                packet_count += 1

        print(f"Total packets processed: {packet_count}")

def main():
    # Paths to the XML files
    ddos_xml_file = 'C:/Users/Shaurya/Downloads/ns3-cybersecurity-simulations/1. NS3.31/ns-allinone-3.31/ns-3.31/IoTAttackSimulation.xml'
    lowrate_tcp_dos_xml_file = 'C:/Users/Shaurya/Downloads/ns3-cybersecurity-simulations/1. NS3.31/ns-allinone-3.31/ns-3.31/IoTSecurityLowRateTCPDoS.xml'

    # Output CSV file paths
    ddos_csv_file = 'C:/Users/Shaurya/Downloads/ns3-cybersecurity-simulations/1. NS3.31/ns-allinone-3.31/ns-3.31/IoTAttackSimulation.csv'
    lowrate_tcp_dos_csv_file = 'C:/Users/Shaurya/Downloads/ns3-cybersecurity-simulations/1. NS3.31/ns-allinone-3.31/ns-3.31/IoTSecurityLowRateTCPDoS.csv'

    # Parse the XML files and convert to CSV
    print("Processing DDoS XML...")
    parse_xml_to_csv(ddos_xml_file, ddos_csv_file, label="DDoS")

    print("Processing LowRateTCPDoS XML...")
    parse_xml_to_csv(lowrate_tcp_dos_xml_file, lowrate_tcp_dos_csv_file, label="LowRateTCPDoS")

    print("Conversion completed. CSV files have been saved.")

if __name__ == "__main__":
    main()
