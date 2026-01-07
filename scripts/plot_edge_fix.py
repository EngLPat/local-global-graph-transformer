"""
Replacement function for plot_edge_connectivity that shows 3 scenarios
"""

# This code should replace the plot_edge_connectivity function starting at line ~276
replacement_code = '''
def plot_edge_connectivity(data, sample_idx, save_path):
    """
    Plot edge connectivity showing 3 scenarios to detect off-by-one errors:
    1. Edges as stored
    2. Edges with indices + 1
    3. Edges with indices - 1
    """
    node_data = data['node_gnn_data'][sample_idx]
    positions = node_data.x.numpy()
    edge_index = node_data.edge_index.numpy()
    
    num_actual_nodes = positions.shape[0]
    x_coords, y_coords = positions[:, 0], positions[:, 1]
    
    # Check for issues
    issues = []
    connected_nodes = np.unique(edge_index.flatten())
    isolated = set(range(num_actual_nodes)) - set(connected_nodes)
    if isolated:
        issues.append(f"⚠ {len(isolated)} isolated nodes (expected for padding)")
    
    # Check off-by-one scenarios
    edge_plus_one = edge_index + 1
    edge_minus_one = edge_index - 1
    
    valid_normal = edge_index.shape[1]
    valid_plus_one = np.sum((edge_plus_one[0] < num_actual_nodes) & (edge_plus_one[1] < num_actual_nodes))
    valid_minus_one = np.sum((edge_minus_one[0] >= 0) & (edge_minus_one[1] >= 0) & 
                            (edge_minus_one[0] < num_actual_nodes) & (edge_minus_one[1] < num_actual_nodes))
    
    # Create 3x2 plot
    fig, axes = plt.subplots(3, 2, figsize=(20, 24))
    fig.suptitle(f'Edge Connectivity Analysis - Sample {sample_idx}', fontsize=16, fontweight='bold')
    
    edge_sample_rate = max(1, edge_index.shape[1] // 2000)
    
    # Row 1: AS STORED
    ax = axes[0, 0]
    ax.scatter(x_coords, y_coords, c='blue', s=15, alpha=0.5, label='Nodes')
    for i in range(0, edge_index.shape[1], edge_sample_rate):
        src, dst = edge_index[:, i]
        if src < num_actual_nodes and dst < num_actual_nodes:
            ax.plot([x_coords[src], x_coords[dst]], [y_coords[src], y_coords[dst]], 
                   'gray', alpha=0.1, linewidth=0.5)
    ax.set_title(f'EDGES AS STORED\\n{valid_normal} edges', fontweight='bold', fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.legend()
    
    # Row 1: +1 SHIFT
    ax = axes[0, 1]
    ax.scatter(x_coords, y_coords, c='blue', s=15, alpha=0.5, label='Nodes')
    for i in range(0, edge_plus_one.shape[1], edge_sample_rate):
        src, dst = edge_plus_one[:, i]
        if src < num_actual_nodes and dst < num_actual_nodes:
            ax.plot([x_coords[src], x_coords[dst]], [y_coords[src], y_coords[dst]], 
                   'red', alpha=0.1, linewidth=0.5)
    ax.set_title(f'EDGES WITH INDICES + 1\\n{valid_plus_one} valid / {edge_plus_one.shape[1]} total', 
                fontweight='bold', fontsize=12, color='darkred')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.legend()
    
    # Row 2: -1 SHIFT
    ax = axes[1, 0]
    ax.scatter(x_coords, y_coords, c='blue', s=15, alpha=0.5, label='Nodes')
    for i in range(0, edge_minus_one.shape[1], edge_sample_rate):
        src, dst = edge_minus_one[:, i]
        if src >= 0 and dst >= 0 and src < num_actual_nodes and dst < num_actual_nodes:
            ax.plot([x_coords[src], x_coords[dst]], [y_coords[src], y_coords[dst]], 
                   'green', alpha=0.1, linewidth=0.5)
    ax.set_title(f'EDGES WITH INDICES - 1\\n{valid_minus_one} valid / {edge_minus_one.shape[1]} total', 
                fontweight='bold', fontsize=12, color='darkgreen')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.legend()
    
    # Row 2: OFF-BY-ONE ANALYSIS
    ax = axes[1, 1]
    ax.axis('off')
    analysis_text = f"""OFF-BY-ONE ERROR ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sample {sample_idx}
Nodes: {num_actual_nodes}

SCENARIO COMPARISON:
▸ AS STORED (gray):
  Valid: {valid_normal}/{edge_index.shape[1]}
  
▸ INDICES + 1 (red):
  Valid: {valid_plus_one}/{edge_plus_one.shape[1]}
  Out-of-bounds: {edge_plus_one.shape[1] - valid_plus_one}
  
▸ INDICES - 1 (green):
  Valid: {valid_minus_one}/{edge_minus_one.shape[1]}
  Invalid: {edge_minus_one.shape[1] - valid_minus_one}

INTERPRETATION:
"""
    if valid_plus_one < edge_index.shape[1] * 0.9:
        analysis_text += "✓ +1 shift breaks connectivity\\n"
    else:
        analysis_text += "⚠ +1 shift looks valid!\\n"
    
    if valid_minus_one < edge_index.shape[1] * 0.9:
        analysis_text += "✓ -1 shift breaks connectivity\\n"
    else:
        analysis_text += "⚠ -1 shift looks valid!\\n"
    
    if valid_plus_one < edge_index.shape[1] * 0.9 and valid_minus_one < edge_index.shape[1] * 0.9:
        analysis_text += "\\n✓ AS STORED is correct\\n"
    else:
        analysis_text += "\\n⚠ Possible off-by-one error!\\n"
    
    ax.text(0.05, 0.95, analysis_text, transform=ax.transAxes, 
           fontsize=11, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Row 3: DIAGNOSTICS
    ax = axes[2, 0]
    ax.axis('off')
    stats_text = f"""GRAPH STATISTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Nodes: {num_actual_nodes}
Edges: {edge_index.shape[1]}
Isolated nodes: {len(isolated)}
(padding expected)

Edge index range:
Min: {edge_index.min()}
Max: {edge_index.max()}
Expected: [0, {num_actual_nodes-1}]

NOTES:
• Isolated padded nodes are OK
• Training should mask them:
  mask = (x.sum(dim=1) != 0)
"""
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
           fontsize=11, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Row 3: ISSUES
    ax = axes[2, 1]
    ax.axis('off')
    issues_text = "DETECTED ISSUES\\n" + "━"*30 + "\\n\\n"
    if issues:
        for issue in issues:
            issues_text += f"{issue}\\n"
    else:
        issues_text += "✓ No critical issues\\n"
    
    ax.text(0.05, 0.95, issues_text, transform=ax.transAxes, 
           fontsize=11, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved edge connectivity plot: {save_path.name}")
    return issues
'''

print(replacement_code)
