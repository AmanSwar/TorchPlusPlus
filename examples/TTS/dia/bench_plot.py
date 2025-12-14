import matplotlib.pyplot as plt
import numpy as np

# Data from benchmarks
input_sizes = [20, 40, 60, 80, 100, 120, 140, 160, 180]

# Original model data
original_time = [0.598, 0.829, 1.096, 1.783, 1.672, 1.805, 3.813, 3.978, 4.134]
original_throughput = [1.672, 1.206, 0.912, 0.561, 0.598, 0.554, 0.262, 0.251, 0.242]

# Optimized model data
optimized_time = [0.023, 0.100, 0.023, 0.116, 0.053, 0.170, 0.048, 0.193, 0.076]
optimized_throughput = [43.482, 10.038, 42.908, 8.643, 18.927, 5.896, 20.898, 5.190, 13.072]

# Calculate speedups
speedup_time = [orig / opt for orig, opt in zip(original_time, optimized_time)]
speedup_throughput = [opt / orig for opt, orig in zip(optimized_throughput, original_throughput)]

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Dia TTS Model: Optimization Performance Analysis', fontsize=16, fontweight='bold')

# 1. Speedup Factor (Bar Chart)
ax1 = plt.subplot(2, 3, 1)
bars = ax1.bar(range(len(input_sizes)), speedup_time, color='#2ecc71', alpha=0.8, edgecolor='black')
ax1.set_xlabel('Input Size', fontweight='bold')
ax1.set_ylabel('Speedup Factor (x)', fontweight='bold')
ax1.set_title('Time Speedup by Input Size', fontweight='bold')
ax1.set_xticks(range(len(input_sizes)))
ax1.set_xticklabels(input_sizes)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.axhline(y=1, color='red', linestyle='--', linewidth=1, label='Baseline')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, speedup_time)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}x', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 2. Time Comparison (Line Chart)
ax2 = plt.subplot(2, 3, 2)
ax2.plot(input_sizes, original_time, marker='o', linewidth=2, markersize=8, 
         label='Original', color='#e74c3c', linestyle='-')
ax2.plot(input_sizes, optimized_time, marker='s', linewidth=2, markersize=8,
         label='Optimized', color='#3498db', linestyle='-')
ax2.set_xlabel('Input Size', fontweight='bold')
ax2.set_ylabel('Time (seconds)', fontweight='bold')
ax2.set_title('Inference Time Comparison', fontweight='bold')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3, linestyle='--')

# 3. Throughput Comparison (Line Chart)
ax3 = plt.subplot(2, 3, 3)
ax3.plot(input_sizes, original_throughput, marker='o', linewidth=2, markersize=8,
         label='Original', color='#e74c3c', linestyle='-')
ax3.plot(input_sizes, optimized_throughput, marker='s', linewidth=2, markersize=8,
         label='Optimized', color='#3498db', linestyle='-')
ax3.set_xlabel('Input Size', fontweight='bold')
ax3.set_ylabel('Throughput (samples/sec)', fontweight='bold')
ax3.set_title('Throughput Comparison', fontweight='bold')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3, linestyle='--')

# 4. Average Speedup Summary (Horizontal Bar)
ax4 = plt.subplot(2, 3, 4)
avg_speedup_time = np.mean(speedup_time)
avg_speedup_throughput = np.mean(speedup_throughput)
categories = ['Time\nReduction', 'Throughput\nIncrease']
values = [avg_speedup_time, avg_speedup_throughput]
colors = ['#2ecc71', '#9b59b6']

bars = ax4.barh(categories, values, color=colors, alpha=0.8, edgecolor='black')
ax4.set_xlabel('Average Speedup Factor (x)', fontweight='bold')
ax4.set_title('Overall Performance Improvement', fontweight='bold')
ax4.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels
for bar, val in zip(bars, values):
    width = bar.get_width()
    ax4.text(width, bar.get_y() + bar.get_height()/2.,
             f'{val:.1f}x', ha='left', va='center', fontweight='bold', fontsize=12, 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

# 5. Time Reduction Percentage (Bar Chart)
ax5 = plt.subplot(2, 3, 5)
time_reduction_pct = [(1 - opt/orig) * 100 for orig, opt in zip(original_time, optimized_time)]
bars = ax5.bar(range(len(input_sizes)), time_reduction_pct, color='#f39c12', alpha=0.8, edgecolor='black')
ax5.set_xlabel('Input Size', fontweight='bold')
ax5.set_ylabel('Time Reduction (%)', fontweight='bold')
ax5.set_title('Percentage Time Saved', fontweight='bold')
ax5.set_xticks(range(len(input_sizes)))
ax5.set_xticklabels(input_sizes)
ax5.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bar, val in zip(bars, time_reduction_pct):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 6. Speedup Heatmap Style Visualization
ax6 = plt.subplot(2, 3, 6)
speedup_data = np.array([speedup_time])
im = ax6.imshow(speedup_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=max(speedup_time))
ax6.set_yticks([0])
ax6.set_yticklabels(['Speedup'])
ax6.set_xticks(range(len(input_sizes)))
ax6.set_xticklabels(input_sizes)
ax6.set_xlabel('Input Size', fontweight='bold')
ax6.set_title('Speedup Intensity Map', fontweight='bold')

# Add text annotations
for i in range(len(input_sizes)):
    text = ax6.text(i, 0, f'{speedup_time[i]:.1f}x',
                   ha="center", va="center", color="black", fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax6, orientation='horizontal', pad=0.1)
cbar.set_label('Speedup Factor', fontweight='bold')

plt.tight_layout()
plt.savefig('tts_speedup_analysis.png', dpi=300, bbox_inches='tight')
print("Graph saved as 'tts_speedup_analysis.png'")
print(f"\nSummary Statistics:")
print(f"Average Time Speedup: {avg_speedup_time:.2f}x")
print(f"Average Throughput Increase: {avg_speedup_throughput:.2f}x")
print(f"Max Time Speedup: {max(speedup_time):.2f}x (at input size {input_sizes[speedup_time.index(max(speedup_time))]})")
print(f"Min Time Speedup: {min(speedup_time):.2f}x (at input size {input_sizes[speedup_time.index(min(speedup_time))]})")
plt.show()