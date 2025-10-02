import matplotlib.pyplot as plt
import numpy as np

# Data
months = ["07/2019", "08/2019", "09/2019", "10/2019", "11/2019"]
searches = [50, 53, 59, 56, 62]
direct = [39, 47, 42, 51, 51]
social_media = [70, 80, 90, 87, 92]

# Plot setup
x = np.arange(len(months))
width = 0.25

fig, ax = plt.subplots(figsize=(8,6))
rects1 = ax.bar(x - width, searches, width, label='Searches', color='skyblue')
rects2 = ax.bar(x, direct, width, label='Direct', color='lightcoral')
rects3 = ax.bar(x + width, social_media, width, label='Social Media', color='gold')

# Labels and titles
ax.set_ylabel("visitors in thousands")
ax.set_xlabel("months")
ax.set_title("Visitors by web traffic sources")
ax.set_xticks(x)
ax.set_xticklabels(months)
ax.legend()

# Bar labels
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0,3),
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)

plt.tight_layout()
plt.show()
