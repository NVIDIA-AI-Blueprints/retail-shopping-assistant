import os
import yaml
import matplotlib.pyplot as plt

NVIDIA_GREEN = '#76B900'
MUTED_RED = '#b97a7a'

# Path to directory with judged YAML files
CONVERSATION = os.environ["TEST_PATH"]
JUDGE_DIR = f'conversations/{CONVERSATION}/judge'

# Load all YAML files
yaml_files = sorted([f for f in os.listdir(JUDGE_DIR) if f.endswith('.yaml')])

# Configure subplots (1 row, N columns or N//cols x cols grid)
n_files = len(yaml_files)
cols = 2  # You can change this to fit your layout
rows = (n_files + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
axes = axes.flatten()  # Flatten in case it's 2D

for i, filename in enumerate(yaml_files):
    with open(os.path.join(JUDGE_DIR, filename), 'r') as f:
        data = yaml.safe_load(f)

    scores = [entry['score'] for entry in data]
    indices = list(range(len(scores)))

    # Determine color for each bar
    colors = [MUTED_RED if score < 3 else NVIDIA_GREEN for score in scores]

    ax = axes[i]
    bars = ax.bar(indices, scores, color=colors)
    ax.set_title(filename, fontsize=10)
    ax.set_ylim(0, 5.5)
    ax.set_xlabel('Query Index')
    ax.set_ylabel('Score')

    # Add horizontal lines at scores 1–5
    for y in range(1, 6):
        ax.axhline(y=y, color='gray', linestyle='--', linewidth=0.5)

    # Annotate bars
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(score),
                ha='center', va='bottom', fontsize=8)

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("LLM-as-judge Scores")
plt.tight_layout()
plt.savefig(f"{JUDGE_DIR}/response_quality.png", dpi=300)
#plt.show()
plt.close()


