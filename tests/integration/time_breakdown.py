import os
import yaml
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CONVERSATION_DIRECTORY = os.environ["TEST_PATH"]

def read_all_timings(directory):
    all_data = {}
    for filename in os.listdir(directory):
        if filename.endswith('.yaml'):
            path = os.path.join(directory, filename)
            with open(path, 'r') as f:
                content = yaml.safe_load(f)
                set_name = content.get("set_name", filename)
                timings = []
                for entry in content.get("results", []):
                    if "timing" in entry:
                        timing = entry["timing"].copy()
                        # Compute additional time not captured in chatter-first_token
                        if "total" in timing and "chatter" in timing and "first_token" in timing:
                            timing["ft_total"] = timing["total"] - (timing["chatter"] - timing["first_token"])
                        else:
                            timing["ft_total"] = 0.0
                        timings.append(timing)
                if timings:
                    all_data[set_name] = timings
    return all_data

def harmonize_timings(timings, all_phases):
    return np.array([[t.get(phase, 0) for phase in all_phases] for t in timings])

def plot_all_averages(data, save_directory):
    # Collect all phases
    all_phases_set = {phase for timings in data.values() for t in timings for phase in t}

    # Ensure 'overhead' and 'total' are always last
    all_phases = sorted([p for p in all_phases_set if p not in {"ft_total", "total"}])
    if "total" in all_phases_set:
        all_phases.append("total")
    if "ft_total" in all_phases_set:
        all_phases.append("ft_total")

    color_map = plt.get_cmap("Set1") 
    colors = {phase: color_map(i / len(all_phases)) for i, phase in enumerate(all_phases)}

    num_files = len(data)
    cols = math.ceil(math.sqrt(num_files))
    rows = math.ceil(num_files / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()

    default_color = "#B0B0B0"
    nvidia_green = "#76B900"
    grey_hatches = ['//', '\\\\', 'x', 'xx', '+', '++', '|','||','..','oo','OO']
    green_hatches = ['//', '\\\\']

    legend_elements = []
    
    # Add grey-patterned legend items for standard phases
    for i, phase in enumerate(all_phases[:-2]):  # All except last two
        legend_elements.append(Patch(
            facecolor=default_color,
            hatch=grey_hatches[i % len(grey_hatches)],
            edgecolor='black',
            label=phase
        ))

    # Add green-patterned legend items for "overhead" and "total"
    for i, phase in enumerate(all_phases[-2:]):
        legend_elements.append(Patch(
            facecolor=nvidia_green,
            hatch=green_hatches[i % len(green_hatches)],
            edgecolor='black',
            label=phase
        ))

    for i, (set_name, timings) in enumerate(data.items()):
        harmonized = harmonize_timings(timings, all_phases)
        means = np.mean(harmonized, axis=0)
        stds = np.std(harmonized, axis=0)

        ax = axes[i]

        for j, phase in enumerate(all_phases):
            if phase in ["ft_total", "total"]:
                color = nvidia_green
                hatch = green_hatches[["ft_total", "total"].index(phase) % len(green_hatches)]
            else:
                color = default_color
                hatch = grey_hatches[j % len(grey_hatches)]
            
            ax.bar(j, means[j], yerr=stds[j], capsize=5, color=color, hatch=hatch, edgecolor='black')

        ax.set_title(set_name)
        ax.set_xticks([])
        if i % cols == 0:
            ax.set_ylabel('Time (seconds)')

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add a legend at the bottom
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0, -0.01, 1, 0.2),  # full-width box
        ncol=len(all_phases),
        mode="expand",
        fontsize=10
    )


    fig.suptitle(f"Average Timing per Phase for '{CONVERSATION_DIRECTORY}' Conversation", fontsize=18)
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])

    # Save to the same directory
    output_path = os.path.join(save_directory, "timing_summary.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to: {output_path}")
    #plt.show()
    plt.close()

# 🔧 Usage
if __name__ == "__main__":
    yaml_directory = f"conversations/{CONVERSATION_DIRECTORY}/results"  

    all_timings = read_all_timings(yaml_directory)
    if all_timings:
        plot_all_averages(all_timings, save_directory=yaml_directory)
    else:
        print("No valid YAML timing data found.")

