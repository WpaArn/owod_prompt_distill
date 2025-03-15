import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow


def draw_mlla_block():
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 8)
    ax.axis('off')

    # Draw the main components of MLLABlock
    ax.add_patch(Rectangle((0, 6), 2, 1, edgecolor='black', facecolor='lightblue', lw=2))
    ax.text(1, 6.5, 'Input', ha='center', va='center', fontsize=10)

    ax.add_patch(Rectangle((0, 4), 2, 1, edgecolor='black', facecolor='lightblue', lw=2))
    ax.text(1, 4.5, 'CPE 1', ha='center', va='center', fontsize=10)

    ax.add_patch(Rectangle((4, 6), 3, 1, edgecolor='black', facecolor='orange', lw=2))
    ax.text(5.5, 6.5, 'Linear Attention', ha='center', va='center', fontsize=10)

    ax.add_patch(Rectangle((4, 4), 3, 1, edgecolor='black', facecolor='lightgreen', lw=2))
    ax.text(5.5, 4.5, 'Activation + Projection', ha='center', va='center', fontsize=10)

    ax.add_patch(Rectangle((8, 5), 2, 1, edgecolor='black', facecolor='lightblue', lw=2))
    ax.text(9, 5.5, 'CPE 2', ha='center', va='center', fontsize=10)

    ax.add_patch(Rectangle((4, 2), 3, 1, edgecolor='black', facecolor='pink', lw=2))
    ax.text(5.5, 2.5, 'MLP', ha='center', va='center', fontsize=10)

    # Draw arrows to represent connections
    arrow_params = dict(facecolor='black', width=0.05, head_width=0.2, length_includes_head=True)

    ax.add_patch(FancyArrow(1, 6, 0, -0.8, **arrow_params))  # Input to CPE 1
    ax.add_patch(FancyArrow(1, 4, 3.5, 0, **arrow_params))  # CPE 1 to Linear Attention
    ax.add_patch(FancyArrow(5.5, 6, 0, -0.8, **arrow_params))  # Linear Attention to Activation + Projection
    ax.add_patch(FancyArrow(5.5, 4, 3, 0.5, **arrow_params))  # Activation to CPE 2
    ax.add_patch(FancyArrow(9, 5, -3.5, -1.5, **arrow_params))  # CPE 2 to MLP
    ax.add_patch(FancyArrow(5.5, 2, 0, -1, **arrow_params))  # MLP to final output

    # Draw a rectangle for the output
    ax.add_patch(Rectangle((4, 0), 3, 1, edgecolor='black', facecolor='lightyellow', lw=2))
    ax.text(5.5, 0.5, 'Output', ha='center', va='center', fontsize=10)

    plt.title('MLLABlock Network Structure')
    # Save the figure to a file
    plt.savefig('mlla_block_structure_diagram.png')
    plt.show()


# Draw the MLLABlock diagram
draw_mlla_block()
