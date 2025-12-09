import csv
import matplotlib.pyplot as plt

def read_results(filename):
    mean_data = {'threads': [], 'speedup': []}
    sdf_data = {'threads': [], 'speedup': []}
    
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            threads = int(row['Threads'])
            speedup = float(row['Speedup'])
            if row['Type'] == 'Mean':
                mean_data['threads'].append(threads)
                mean_data['speedup'].append(speedup)
            elif row['Type'] == 'SDF':
                sdf_data['threads'].append(threads)
                sdf_data['speedup'].append(speedup)
                
    return mean_data, sdf_data

def plot_performance(mean_data, sdf_data):
    plt.figure(figsize=(10, 6))
    
    plt.plot(mean_data['threads'], mean_data['speedup'], marker='o', label='Mean Computation')
    plt.plot(sdf_data['threads'], sdf_data['speedup'], marker='s', label='SDF Computation')
    
    # Plot ideal linear speedup
    max_threads = max(max(mean_data['threads']), max(sdf_data['threads']))
    plt.plot([1, max_threads], [1, max_threads], 'k--', label='Ideal Linear Speedup', alpha=0.5)
    
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup')
    plt.title('Performance Speedup: Threads vs Speedup')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('performance_graph.png')
    print("Graph saved to performance_graph.png")

if __name__ == "__main__":
    mean_data, sdf_data = read_results('results.csv')
    plot_performance(mean_data, sdf_data)
