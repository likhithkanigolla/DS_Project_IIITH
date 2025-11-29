"""Matplotlib visualization and animation of MST building process."""
import os
from typing import List, Tuple
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def save_iteration_tables(iteration_snapshots: List[List[Tuple[int,int,float]]],
                          iteration_unions: List[List[Tuple[int,int]]],
                          component_counts: List[int],
                          out_dir: str):
    """Generate per-iteration HTML tables showing algorithm steps."""
    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Borůvka MST - Iteration Details</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
h2 { color: #3498db; margin-top: 30px; background-color: #e8f4f8; padding: 10px; border-radius: 5px; }
h3 { color: #2c3e50; margin-top: 20px; }
table { border-collapse: collapse; width: 100%; margin: 15px 0; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
th { background-color: #3498db; color: white; font-weight: bold; }
tr:nth-child(even) { background-color: #f9f9f9; }
tr:hover { background-color: #e8f4f8; }
.new-edge { background-color: #fff3cd; font-weight: bold; border-left: 4px solid #ffc107; }
.stats { background-color: #e8f4f8; padding: 15px; border-left: 4px solid #3498db; margin: 20px 0; font-size: 16px; }
.stats strong { color: #2c3e50; }
.algorithm-step { background-color: #d4edda; padding: 10px; border-left: 4px solid #28a745; margin: 10px 0; }
</style>
</head>
<body>
<h1>Borůvka MST Algorithm - Detailed Execution Trace</h1>
<div class="algorithm-step">
<strong>Algorithm Steps:</strong><br>
A) Each machine finds minimum weight edges for its local components<br>
B) All machines send candidates to rank 0 (Gather)<br>
C) Rank 0 selects global minimum edge per component (using DSU roots)<br>
D) Rank 0 broadcasts union operations to all machines<br>
E) Each machine applies unions locally
</div>
""")
    
    prev_edges = set()
    for i, snapshot in enumerate(iteration_snapshots):
        iter_no = i + 1
        current_edges = {tuple(sorted((u, v))) for (u, v, w) in snapshot}
        new_edges = current_edges - prev_edges
        unions = iteration_unions[i] if i < len(iteration_unions) else []
        comps = component_counts[i] if i < len(component_counts) else '?'
        
        html_parts.append(f"<h2>Iteration {iter_no}</h2>")
        html_parts.append(f'<div class="stats">')
        html_parts.append(f"<strong>Components Remaining:</strong> {comps} | ")
        html_parts.append(f"<strong>MST Edges So Far:</strong> {len(snapshot)} | ")
        html_parts.append(f"<strong>New Edges Added:</strong> {len(new_edges)}")
        html_parts.append(f"</div>")
        
        # Union operations table (Steps C-D)
        if unions:
            html_parts.append("<h3>Steps C-D: Union Operations (Component Merges)</h3>")
            html_parts.append("<table>")
            html_parts.append("<tr><th>Operation #</th><th>Component Root A</th><th>Component Root B</th><th>Action</th></tr>")
            for idx, (a, b) in enumerate(unions, 1):
                html_parts.append(f"<tr><td>{idx}</td><td>{a}</td><td>{b}</td><td>Union(A, B) - Merge components</td></tr>")
            html_parts.append("</table>")
        
        # New edges table (Steps A-B)
        if new_edges:
            html_parts.append("<h3>Steps A-B: Edges Selected This Iteration</h3>")
            html_parts.append("<table>")
            html_parts.append("<tr><th>Edge #</th><th>Source Node (u)</th><th>Target Node (v)</th><th>Weight</th><th>Status</th></tr>")
            edge_num = 1
            for (u, v, w) in snapshot:
                key = tuple(sorted((u, v)))
                if key in new_edges:
                    html_parts.append(f'<tr class="new-edge"><td>{edge_num}</td><td>{u}</td><td>{v}</td><td>{w:.6f}</td><td>✓ ADDED</td></tr>')
                    edge_num += 1
            html_parts.append("</table>")
        
        # All edges accumulated (full MST so far)
        html_parts.append(f"<h3>Complete MST Edge List After Iteration {iter_no}</h3>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Edge #</th><th>Source (u)</th><th>Target (v)</th><th>Weight</th><th>Added in Iteration</th></tr>")
        
        # To show when each edge was added, we need to track this
        edge_to_iter = {}
        temp_prev = set()
        for j, temp_snapshot in enumerate(iteration_snapshots[:i+1], 1):
            temp_current = {tuple(sorted((u, v))): w for (u, v, w) in temp_snapshot}
            for key in temp_current:
                if key not in temp_prev:
                    edge_to_iter[key] = j
            temp_prev = set(temp_current.keys())
        
        for idx, (u, v, w) in enumerate(snapshot, 1):
            key = tuple(sorted((u, v)))
            added_iter = edge_to_iter.get(key, '?')
            row_class = 'class="new-edge"' if key in new_edges else ''
            html_parts.append(f"<tr {row_class}><td>{idx}</td><td>{u}</td><td>{v}</td><td>{w:.6f}</td><td>Iteration {added_iter}</td></tr>")
        html_parts.append("</table>")
        
        prev_edges = current_edges
    
    # Final summary
    if iteration_snapshots:
        final_mst = iteration_snapshots[-1]
        total_weight = sum(w for (u, v, w) in final_mst)
        html_parts.append("<h2>Final MST Summary</h2>")
        html_parts.append(f'<div class="stats">')
        html_parts.append(f"<strong>Total Edges:</strong> {len(final_mst)} | ")
        html_parts.append(f"<strong>Total Weight:</strong> {total_weight:.6f} | ")
        html_parts.append(f"<strong>Total Iterations:</strong> {len(iteration_snapshots)}")
        html_parts.append(f"</div>")
    
    html_parts.append("</body></html>")
    
    html_path = os.path.join(out_dir, 'iteration_details.html')
    with open(html_path, 'w') as f:
        f.write('\n'.join(html_parts))
    return html_path


def save_animation(num_nodes: int,
                   edges: List[Tuple[int,int,float]],
                   iteration_snapshots: List[List[Tuple[int,int,float]]],
                   out_dir: str,
                   title: str = "Boruvka MST - Iterations",
                   size: int = 1,
                   highlight_new: bool = True,
                   component_counts: List[int] | None = None,
                   max_frames: int | None = None):
    os.makedirs(out_dir, exist_ok=True)

    # Basic layout: place nodes on a circle
    import math
    angles = [2*math.pi*i/num_nodes for i in range(num_nodes)]
    coords = [(math.cos(a), math.sin(a)) for a in angles]

    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_title(title)
    ax.set_axis_off()

    # assign colors per rank for nodes using contiguous partitioning by rank
    per = num_nodes // size if size > 0 else num_nodes
    extras = num_nodes % size if size > 0 else 0
    ranges = []
    start = 0
    for i in range(size):
        cnt = per + (1 if i < extras else 0)
        ranges.append((start, start+cnt))
        start += cnt
    node_color = ['black'] * num_nodes
    palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    for r_idx, (s, e) in enumerate(ranges):
        color = palette[r_idx % len(palette)]
        for u in range(s, e):
            node_color[u] = color

    def draw_frame(mst_edges, iteration=None):
        ax.clear()
        ttl = title if iteration is None else f"{title} (iter {iteration})"
        ax.set_title(ttl)
        ax.set_axis_off()
        # draw all edges faint
        for (u, v, w) in edges:
            x1, y1 = coords[u]
            x2, y2 = coords[v]
            ax.plot([x1, x2], [y1, y2], color='lightgray', linewidth=0.3, zorder=1)
        # draw mst edges thinner
        for (u, v, w) in mst_edges:
            x1, y1 = coords[u]
            x2, y2 = coords[v]
            ax.plot([x1, x2], [y1, y2], color='tab:blue', linewidth=0.5, zorder=2)
        # draw nodes
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        ax.scatter(xs, ys, s=20, c=node_color, zorder=3)
        
        # Add legend showing rank/machine color coding
        from matplotlib.patches import Patch
        legend_elements = []
        for r_idx in range(min(size, len(palette))):
            color = palette[r_idx % len(palette)]
            s, e = ranges[r_idx] if r_idx < len(ranges) else (0, 0)
            legend_elements.append(Patch(facecolor=color, label=f'Rank {r_idx} (nodes {s}-{e-1})'))
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)

    # Determine which iterations to render (sampling if needed)
    total_iters = len(iteration_snapshots)
    indices = list(range(total_iters))
    if max_frames and max_frames > 0 and total_iters > max_frames:
        step = (total_iters - 1) / (max_frames - 1)
        sampled = []
        for k in range(max_frames):
            sampled.append(round(k * step))
        indices = sorted(set(sampled))

    # Save frames as PNGs
    frame_paths = []
    # Save initial graph image
    draw_frame([], iteration=0)
    fig.savefig(os.path.join(out_dir, "initial_graph.png"), dpi=150, bbox_inches='tight')

    prev_set = set()
    for idx in indices:
        mst_edges = iteration_snapshots[idx]
        # Handle both formats: [(u,v,w)] and [[u,v]]
        if mst_edges and len(mst_edges[0]) == 3:
            # Format: [(u, v, w)]
            current_set = {tuple(sorted((u, v))) for (u, v, w) in mst_edges}
        else:
            # Format: [[u, v]] or [(u, v)]
            current_set = {tuple(sorted(edge)) for edge in mst_edges}
        new_set = current_set - prev_set if highlight_new else current_set
        # Redraw with differentiation
        ax.clear()
        iter_no = idx + 1
        ttl = f"{title} (iter {iter_no})"
        ax.set_title(ttl)
        ax.set_axis_off()
        for (u, v, w) in edges:
            x1, y1 = coords[u]; x2, y2 = coords[v]
            ax.plot([x1, x2], [y1, y2], color='lightgray', linewidth=0.3, zorder=1)
        
        # Determine which rank owns each node for edge coloring
        def get_rank_for_node(node):
            for r_idx, (s, e) in enumerate(ranges):
                if s <= node < e:
                    return r_idx
            return 0
        
        # Old MST edges: colored arrows by source rank
        if highlight_new:
            for (u, v, w) in mst_edges:
                key = tuple(sorted((u, v)))
                if key in new_set:
                    continue
                rank_owner = get_rank_for_node(u)
                edge_color = palette[rank_owner % len(palette)]
                x1, y1 = coords[u]; x2, y2 = coords[v]
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->', color=edge_color, lw=0.9, alpha=0.6, shrinkA=5, shrinkB=5),
                           zorder=2)
            # New edges: thicker arrows in source rank color
            for (u, v, w) in mst_edges:
                key = tuple(sorted((u, v)))
                if key not in new_set:
                    continue
                rank_owner = get_rank_for_node(u)
                edge_color = palette[rank_owner % len(palette)]
                x1, y1 = coords[u]; x2, y2 = coords[v]
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->', color=edge_color, lw=2.0, alpha=1.0, shrinkA=5, shrinkB=5),
                           zorder=3)
        else:
            for (u, v, w) in mst_edges:
                rank_owner = get_rank_for_node(u)
                edge_color = palette[rank_owner % len(palette)]
                x1, y1 = coords[u]; x2, y2 = coords[v]
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->', color=edge_color, lw=1.2, shrinkA=5, shrinkB=5),
                           zorder=2)
        # Nodes
        xs = [c[0] for c in coords]; ys = [c[1] for c in coords]
        ax.scatter(xs, ys, s=20, c=node_color, zorder=4)
        # Legend (rank ownership)
        from matplotlib.patches import Patch
        legend_elements = []
        for r_idx in range(min(size, len(palette))):
            color = palette[r_idx % len(palette)]
            s, e = ranges[r_idx] if r_idx < len(ranges) else (0, 0)
            legend_elements.append(Patch(facecolor=color, label=f"Rank {r_idx} {s}-{e-1}"))
        ax.legend(handles=legend_elements, loc='upper right', fontsize=6, framealpha=0.9)
        # Annotation box
        comps = component_counts[idx] if component_counts and idx < len(component_counts) else '?'
        ax.text(0.02, 0.98,
                f"iter {iter_no}\nMST edges {len(mst_edges)}\ncomponents {comps}",
                transform=ax.transAxes, va='top', ha='left', fontsize=7,
                bbox=dict(boxstyle='round', fc='white', alpha=0.75))
        frame_path = os.path.join(out_dir, f"frame_{iter_no:03d}.png")
        fig.savefig(frame_path, dpi=120, bbox_inches='tight')
        frame_paths.append(frame_path)
        prev_set = current_set

    # Save the final static MST image with FINAL label
    if iteration_snapshots:
        final_mst = iteration_snapshots[-1]
        ax.clear()
        ax.set_title(f"{title} - FINAL MST", fontsize=12, fontweight='bold')
        ax.set_axis_off()
        # Draw input graph faint
        for (u, v, w) in edges:
            x1, y1 = coords[u]; x2, y2 = coords[v]
            ax.plot([x1, x2], [y1, y2], color='lightgray', linewidth=0.3, zorder=1)
        # Draw final MST edges colored by rank
        def get_rank_for_node(node):
            for r_idx, (s, e) in enumerate(ranges):
                if s <= node < e:
                    return r_idx
            return 0
        for (u, v, w) in final_mst:
            rank_owner = get_rank_for_node(u)
            edge_color = palette[rank_owner % len(palette)]
            x1, y1 = coords[u]; x2, y2 = coords[v]
            ax.plot([x1, x2], [y1, y2], color=edge_color, linewidth=1.5, zorder=2)
        xs = [c[0] for c in coords]; ys = [c[1] for c in coords]
        ax.scatter(xs, ys, s=20, c=node_color, zorder=3)
        # Summary annotation
        total_weight = sum(w for (u, v, w) in final_mst)
        ax.text(0.5, 0.02,
                f"FINAL: {len(final_mst)} edges | Total weight: {total_weight:.2f}",
                transform=ax.transAxes, va='bottom', ha='center', fontsize=8,
                bbox={'boxstyle': 'round', 'fc': 'yellow', 'alpha': 0.9}, fontweight='bold')
        fig.savefig(os.path.join(out_dir, "final_mst.png"), dpi=150, bbox_inches='tight')

    plt.close(fig)
    return frame_paths


def build_gif(frame_paths: List[str], gif_path: str, duration: float = 0.6):
    """Build a GIF from saved frame image paths."""
    images = []
    for p in frame_paths:
        try:
            images.append(imageio.imread(p))
        except Exception:
            pass
    if images:
        # Use fps (frames per second) for reliable timing: fps = 1/duration
        fps = 1.0 / duration if duration > 0 else 1.0
        imageio.mimsave(gif_path, images, fps=fps, loop=0)
    return gif_path


def save_growth_chart(component_counts: List[int], mst_sizes: List[int], out_dir: str):
    """Save a growth curve showing MST edge accumulation and component count decay."""
    if not component_counts or not mst_sizes:
        return None
    import matplotlib.pyplot as plt
    iters = list(range(1, len(mst_sizes)+1))
    fig, ax1 = plt.subplots(figsize=(6,4))
    ax1.plot(iters, mst_sizes, color='#1f77b4', label='MST edges')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('MST edge count', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax2 = ax1.twinx()
    ax2.plot(iters, component_counts[:len(mst_sizes)], color='#d62728', label='Components')
    ax2.set_ylabel('Components', color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    fig.suptitle('Boruvka Progress')
    # Combined legend
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    fig.legend(lines, labels, loc='upper center', ncol=2)
    path = os.path.join(out_dir, 'growth_curve.png')
    fig.tight_layout(rect=[0,0,1,0.92])
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path



