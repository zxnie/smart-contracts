import pickle
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
import seaborn as sns

# import plotly.graph_objects as go

# %%
'''
Throughput
'''
simulation_time = 1000000
num_edge_node_list = [4, 4, 4, 5, 5, 6, 6]
num_robot_node_list = [9, 13, 19, 23, 29, 31, 37]
total_num_nodes_list = []
average_throughput_list = []
for num_edge_nodes, num_robot_nodes in zip(num_edge_node_list, num_robot_node_list):
    fn = f'{num_edge_nodes}edge_{num_robot_nodes}robots_{simulation_time}ms'

    with open(f'results_{fn}.pkl', 'rb') as f:
        results = pickle.load(f)

    history = results['fl_history']
    wireless_stats = results['wireless_stats']

    data = wireless_stats['throughput_list']
    x = wireless_stats['t']

    total_num_nodes_list.append(num_edge_nodes + num_robot_nodes)
    average_throughput_list.append(sum(data) / (x[-1] / 100))

df_throughput = pd.DataFrame({'total_nodes': total_num_nodes_list, 'throughput': average_throughput_list})
sns.lineplot(data=df_throughput, x='total_nodes', y='throughput')

# %%
x = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
# unit_pixel = 206
# y = np.array((89, 103, 117, 135, 154, 177, 204, 220, 275, 323)) + unit_pixel
unit_pixel = 208
y = np.array((159, 181, 204, 238, 278, 329, 402, 509, 681, 996)) + 2 * unit_pixel
y = y / unit_pixel * 0.002

y_interpolate = interpolate.interp1d(x, y, kind='cubic', fill_value='extrapolate')

average_delay_list = []
for throughput in average_throughput_list:
    average_delay_list.append(float(y_interpolate(throughput + 20)))

average_delay_list = np.array(average_delay_list) * 1000

print(average_delay_list)

# %%
'''
Latency
'''
simulation_time = 100000
num_edge_node_list = [4, 4, 4, 5, 5, 6, 6]
num_robot_node_list = [9, 13, 19, 23, 29, 31, 37]
total_num_nodes_list = []
average_latency_list = []
for num_edge_nodes, num_robot_nodes in zip(num_edge_node_list, num_robot_node_list):
    fn = f'{num_edge_nodes}edge_{num_robot_nodes}robots_{simulation_time}ms_latency'
    with open(f'results_{fn}.pkl', 'rb') as f:
        results = pickle.load(f)
    data = results['wireless_stats']['latency_list']
    total_num_nodes_list.append(num_edge_nodes + num_robot_nodes)
    average_latency = sum(data) / len(data)
    average_latency_list.append(average_latency)

df_latency = pd.DataFrame({'total_nodes': total_num_nodes_list, 'latency': average_latency_list})
sns.lineplot(data=df_latency, x='total_nodes', y='latency')

# %%
'''
Analysis plot
'''
fn = f'analysis'

with open(f'results_{fn}.pkl', 'rb') as f:
    results_analysis = pickle.load(f)


def interpolate_df(df):
    # Assuming the DataFrame is named df
    total_nodes = df['total_nodes'].iloc[0]  # All values are the same, so take the first one
    # Create new x values
    new_x = np.linspace(0, 1, 50)
    # Interpolate analytical
    f_analytical = interpolate.interp1d(df['x'], df['analytical'], kind='cubic', fill_value='extrapolate')
    new_analytical = f_analytical(new_x)
    # Interpolate simulation
    f_simulation = interpolate.interp1d(df['x'], df['simulation'], kind='cubic', fill_value='extrapolate')
    new_simulation = f_simulation(new_x)
    # Create new DataFrame
    new_df = pd.DataFrame({
        'total_nodes': [total_nodes] * 50,
        'x': new_x,
        'analytical': new_analytical,
        'simulation': new_simulation
    })
    return new_df


df_list = []
for node_nums, result in results_analysis.items():
    node_num_list = node_nums.split(',')
    node_num = sum([int(i) for i in node_num_list])
    total_nodes = np.ones(len(result['x_list'])) * node_num
    x = np.array(result['x_list']) / len(result['x_list'])
    df = pd.DataFrame({
        'total_nodes': total_nodes,
        'x': x,
        'analytical': result['P1_list'],
        'simulation': result['P2_list'],
    })
    df_new = interpolate_df(df)
    df_list.append(df_new)
df = pd.concat(df_list, axis=0)

# %%
# Pivot your data to get a grid
df_pivot = df.pivot(index='x', columns='total_nodes', values='analytical')
# Generate x, y, z values for the plot
X, Y = np.meshgrid(df_pivot.columns, df_pivot.index)
Z = df_pivot.values
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
# Plot the surface
ax.plot_surface(X, Y, Z)
ax.set_xlabel('Total nodes')
ax.set_ylabel('x')
ax.set_zlabel('Probability')
plt.tight_layout()
plt.show()

# %%
# Pivot your data to get a grid
df_pivot = df.pivot(index='x', columns='total_nodes', values='simulation')
# Generate x, y, z values for the plot
X, Y = np.meshgrid(df_pivot.columns, df_pivot.index)
Z = df_pivot.values
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
# Plot the surface
ax.plot_surface(X, Y, Z)
ax.set_xlabel('Total nodes')
ax.set_ylabel('x')
ax.set_zlabel('Probability')
plt.tight_layout()
plt.show()

# %%
# Filter data where both 'analytical' and 'simulation' are >= 0.99
filtered_df = df[(df['analytical'] >= 0.99) & (df['simulation'] >= 0.99)]

# Group by 'total_nodes' and calculate the mean of 'x'
result = filtered_df.groupby('total_nodes')['x'].min().reset_index()

# Print the result
print(result)

print(result['x'].mean())


# %%
def get_high_low_avg(data, window_size=500):
    avg = []
    high = []
    low = []
    for i in range(len(data)):
        data_segment = np.array(data[i:i + window_size])
        non_zero_indices = np.nonzero(data_segment)[0]
        if len(non_zero_indices) == 0:
            break
        data_segment_non_zero = data_segment[non_zero_indices]
        avg.append(np.mean(data_segment_non_zero))
        high.append(np.max(data_segment_non_zero))
        low.append(np.min(data_segment_non_zero))
    return avg, high, low


# %%

fig, ax = plt.subplots()
data = wireless_stats['throughput_list']
x = wireless_stats['t']
ax.plot(x, data)
# avg, high, low = get_high_low_avg(data, 5000)
# ax.plot(x[:len(avg)], avg)
# ax.fill_between(x[:len(avg)], high, low, alpha=0.2)
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Messages / millisecond')
ax.set_title('Throughput')
fig.show()

print(f'Average throughput: {sum(data) / (x[-1] / 10)} messages / time frame (10 ms)')

# %%
'''
Latency
'''
fig, ax = plt.subplots()
data = wireless_stats['latency_list']
x = wireless_stats['t']
ax.plot(x, data)
avg, high, low = get_high_low_avg(data, 5000)
ax.plot(x[:len(avg)], avg)
ax.fill_between(x[:len(avg)], high, low, alpha=0.2)
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Millisecond / message')
ax.set_title('Latency')
fig.show()

# %%
'''
FL plot
'''

num_edge_nodes = 4
num_robot_nodes = 9
simulation_time = 100000

fn = f'{num_edge_nodes}edge_{num_robot_nodes}robots_{simulation_time}ms'

with open(f'results_{fn}.pkl', 'rb') as f:
    results = pickle.load(f)

history = results['fl_history']
wireless_stats = results['wireless_stats']

fig, ax = plt.subplots()
train_loss = history['train_loss'][1:]
val_loss = history['valid_loss'][1:]
ax.plot([i + 1 for i in range(len(train_loss))], train_loss,
        color='r', label='train loss')
ax.plot([i + 1 for i in range(len(val_loss))], val_loss,
        color='b', label='validation loss')
ymin = min(history['train_loss'][1:] + history['valid_loss'][1:])
ymax = max(history['train_loss'][1:] + history['valid_loss'][1:])
ax.vlines(x=history['x'],
          ymin=ymin, ymax=ymax,
          color='g', linestyle='--', label='dataset change')
ax.legend()
ax.set_title('Training history')
fig.show()

#%%
