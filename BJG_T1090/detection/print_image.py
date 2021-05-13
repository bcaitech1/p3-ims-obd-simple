import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns


category_names = ['Background', 'UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal', 
                  'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']    
    
def print_image(images, masks, image_infos, batch_size):    
    cm = 'inferno' # tab10
    nrows = batch_size//2
    ncols = 4
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, batch_size*1.5))

    idx = 0
    for i in range(nrows):
        for j in [0, 2]:
            categories = np.unique(masks[idx])
            
            # Original image
            axes[i][j].imshow(images[idx].permute([1,2,0]))
            axes[i][j].grid(False)
            axes[i][j].set_title("Original image : {}".format(image_infos[idx]['file_name']), fontsize = 15)
                
            # Predicted
            img = axes[i][j+1].imshow(masks[idx], cmap=plt.get_cmap(cm))
            axes[i][j+1].grid(False)

            divider = make_axes_locatable(axes[i][j+1])
            cax = divider.append_axes("right", size="5%", pad=0.05)

            cbar = fig.colorbar(img, cax=cax, ax=axes[i][j+1], ticks=[int(i) for i in list(categories)])
            cbar.ax.set_yticklabels([category_names[int(i)] for i in list(categories)])

            axes[i][j+1].set_title("Predicted", fontsize = 15)
            
            # 사각형 그리기 https://www.delftstack.com/ko/howto/matplotlib/how-to-draw-rectangle-on-image-in-matplotlib/
            for xmin, ymin, xmax, ymax in image_infos[idx]['bbox']:
                x_bottom_left = xmin
                y_bottom_left = ymin
                width = xmax-xmin
                height = ymax-ymin
                axes[i][j+1].add_patch(
                    patches.Rectangle(
                        (x_bottom_left, y_bottom_left),
                        width,
                        height,
                        fill=False,
                        color='lime',
#                         lw=3
                    )
                )

            idx += 1

    plt.show()


def compare_augmenation(batch_size, ori_images, image_infos, aug_images1, aug_images2, aug_images3, aug1, aug2, aug3):
    nrows = batch_size
    ncols = 4
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, batch_size*6))

    for i in range(nrows):
        # Original image
        axes[i][0].imshow(ori_images[i].permute([1,2,0]), aspect='auto')
        axes[i][0].grid(False)
        axes[i][0].set_title(f"{image_infos[i]['file_name']}", fontsize = 15)
        axes[i][0].set_xticks([])
        axes[i][0].set_yticks([])
        
        for xmin, ymin, xmax, ymax in image_infos[i]['bbox']:
                x_bottom_left = xmin
                y_bottom_left = ymin
                width = xmax-xmin
                height = ymax-ymin
                axes[i][0].add_patch(
                    patches.Rectangle(
                        (x_bottom_left, y_bottom_left),
                        width,
                        height,
                        fill=False,
                        color='lime',
#                         lw=3
                    )
                )

        # Augmentational image1
        axes[i][1].imshow(aug_images1[i].permute([1,2,0]), aspect='auto')
        axes[i][1].grid(False)
        axes[i][1].set_title(f"{aug1}", fontsize = 15)
        axes[i][1].set_xticks([])
        axes[i][1].set_yticks([])
        
        # Augmentational image2
        axes[i][2].imshow(aug_images2[i].permute([1,2,0]), aspect='auto')
        axes[i][2].grid(False)
        axes[i][2].set_title(f"{aug2}", fontsize = 15)
        axes[i][2].set_xticks([])
        axes[i][2].set_yticks([])
        
        # Augmentational image3
        axes[i][3].imshow(aug_images3[i].permute([1,2,0]), aspect='auto')
        axes[i][3].grid(False)
        axes[i][3].set_title(f"{aug3}", fontsize = 15)
        axes[i][3].set_xticks([])
        axes[i][3].set_yticks([])


    plt.show()
