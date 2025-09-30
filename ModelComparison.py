import matplotlib.pyplot as plt
import numpy as np

# --- Plotting Comparison ---

# Assumes you have these variables from running all three scripts:
# top1_acc, top5_acc (from CLIP script)
# top1_acc_blip, top5_acc_blip (from BLIP script)
# top1_acc_siglip, top5_acc_siglip (from SigLIP script)

# Example data (replace with your actual results)
top1_acc = 92.50 
top5_acc = 98.00
top1_acc_blip = 94.20
top5_acc_blip = 99.10
top1_acc_siglip = 95.80
top5_acc_siglip = 99.50
# ---

models = ["CLIP", "BLIP", "SigLIP"]
top1_scores = [top1_acc, top1_acc_blip, top1_acc_siglip]
top5_scores = [top5_acc, top5_acc_blip, top5_acc_siglip]

x = np.arange(len(models))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 7))
rects1 = ax.bar(x - width/2, top1_scores, width, label='Top-1 Accuracy')
rects2 = ax.bar(x + width/2, top5_scores, width, label='Top-5 Accuracy')

# Add some text for labels, title and axes ticks
ax.set_ylabel('Accuracy (%)')
ax.set_title('Image Retrieval Accuracy: CLIP vs. BLIP vs. SigLIP')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 105)
ax.legend()

# Attach a text label above each bar
ax.bar_label(rects1, padding=3, fmt='%.1f')
ax.bar_label(rects2, padding=3, fmt='%.1f')

fig.tight_layout()
plt.show()