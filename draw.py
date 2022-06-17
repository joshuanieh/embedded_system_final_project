import matplotlib.pyplot as plt
img = plt.imread("training_data/2789_cam-image_array_.jpg")
fig, ax = plt.subplots()
ax.imshow(img, extent=[0, 160, 0, 120])
print(fig.get_size_inches()*fig.dpi)
x = range(int(160/10) + 1)
y = range(int(120/10) + 1)
for i in y:
    ax.plot([10*k for k in x], [10*i for j in x], '--', linewidth=1, color='firebrick')

for i in x:
    ax.plot([10*i for j in y], [10*k for k in y], '--', linewidth=1, color='firebrick')
#plt.show()
plt.savefig("cor.png")
