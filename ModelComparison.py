import matplotlib.pyplot as plt
import numpy as np

# --- Plotting Comparison ---

#actual result
top1_acc_clip = 55.00 
top5_acc_clip = 95.00
top1_acc_blip = 5.00
top5_acc_blip = 30.00
top1_acc_siglip = 5.00
top5_acc_siglip = 35.00
top1_acc_blip2 = 0.00  # Update this after running BLIP-2
top5_acc_blip2 = 0.00  # Update this after running BLIP-2
# ---

models = ["CLIP", "BLIP", "SigLIP", "BLIP-2"]
top1_scores = [top1_acc_clip, top1_acc_blip, top1_acc_siglip, top1_acc_blip2]
top5_scores = [top5_acc_clip, top5_acc_blip, top5_acc_siglip, top5_acc_blip2]

x = np.arange(len(models))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 7))
rects1 = ax.bar(x - width/2, top1_scores, width, label='Top-1 Accuracy')
rects2 = ax.bar(x + width/2, top5_scores, width, label='Top-5 Accuracy')

# Add some text for labels, title and axes ticks
ax.set_ylabel('Accuracy (%)')
ax.set_title('Image Retrieval Accuracy: CLIP vs. BLIP vs. SigLIP vs. BLIP-2')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 105)
ax.legend()

# Attach a text label above each bar
ax.bar_label(rects1, padding=3, fmt='%.1f')
ax.bar_label(rects2, padding=3, fmt='%.1f')

fig.tight_layout()
plt.show()