import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = r"C:\Users\Shaurya\Downloads\FuzzAIoT\data\CombinedTraffic.csv"
df = pd.read_csv(file_path)

# Convert the timestamp to a datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')

# Plot 1: Distribution of Packet Sizes
plt.figure(figsize=(10, 6))
plt.hist(df['PacketSize'], bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Packet Sizes')
plt.xlabel('Packet Size')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Plot 2: Number of Packets Over Time
plt.figure(figsize=(10, 6))
df.set_index('Timestamp').resample('1T').size().plot(color='purple')
plt.title('Number of Packets Over Time')
plt.xlabel('Time')
plt.ylabel('Number of Packets')
plt.grid(True)
plt.show()

# Plot 3: Class Distribution (Benign vs. DDoS)
plt.figure(figsize=(8, 6))
df['Label'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Class Distribution: Benign vs. DDoS')
plt.xlabel('Class')
plt.ylabel('Number of Packets')
plt.xticks(rotation=0)
plt.grid(True)
plt.show()
