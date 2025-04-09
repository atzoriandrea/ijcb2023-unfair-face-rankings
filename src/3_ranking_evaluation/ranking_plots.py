import numpy as np
import pandas as pd  # Data manipulation library
import matplotlib.pyplot as plt  # Plotting library
from config import *  # Custom configuration importing dataset paths and model names
import cv2  # OpenCV for image processing
import random  # Random selection utilities

# Get datasets names defined in config
datasets = models_data_HR_HR.keys()

# Function to recursively fetch full file paths for pickle files
def get_files_full_path(rootdir):
    import os
    paths = []
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith(".pkl"):
                paths.append(os.path.join(root, file))
    return paths

# Iterate over dataset settings (here only 'HR-HR')
for s_i, setting in enumerate(["HR-HR"]):
    for d_i, dataset in enumerate(datasets):
        model_ranking = []

        # Load pickle files containing rankings for the current dataset
        files = get_files_full_path("/media/Workspace/Ranking/IJCB23/" + setting + "/" + dataset)

        # Load rankings for each model based on their names
        for model_name in model_names:
            current = [x for x in files if model_name in x][0]
            model_ranking.append(pd.read_pickle(current))

        # Load the list of file paths from the test set
        info_file = info[dataset]["test_set_file"]
        with open(info_file, "r") as ifp:
            fileslist = ifp.readlines()

        # Identify unique demographic groups present in the dataset
        groups = sorted(np.unique(model_ranking[0]["group"]))

        # Iterate through each model to visualize rankings
        for m_i, model in enumerate(model_names):
            plt.rcParams.update({'font.size': 18})
            img_path = "output/" + dataset + "/" + model + "_paper.png"

            # Skip image generation if file already exists
            if os.path.exists(img_path):
                print("Skipping %s -- file already exists" % img_path)
            else:
                # Load embeddings to fetch identity classes
                npy = np.load(models_data_HR_HR[dataset][m_i])[:, 512]

                # Create a mapping from class ID to file paths
                info_dict = {}
                for i, line in enumerate(fileslist[:npy.shape[0]]):
                    p, c = line.strip().split(" ")
                    if str(int(npy[i])) in info_dict.keys():
                        info_dict[str(int(npy[i]))].append(p)
                    else:
                        info_dict[str(int(npy[i]))] = [p]

                # Select random samples per group
                n_samples = 3 if dataset != "RFW" else 2
                samples = model_ranking[m_i][model_ranking[m_i]['probe'] == 1].groupby("group").sample(n_samples, random_state=42 if dataset != "RFW" else 5483)

                ranks = {}
                for index, sample in samples.iterrows():
                    matches = sample.matches
                    cls = sample["class"]
                    id_grps = []
                    id_cls = []

                    # Extract group and class info for each match
                    for m in matches:
                        id_grps.append(model_ranking[m_i].loc[m - 1]["group"])
                        id_cls.append(model_ranking[m_i].loc[m - 1]["class"])

                    ranks[sample["path"]] = {"class": cls, "groups": id_grps, "ranked_classes": id_cls}

                rows = n_samples * len(groups)
                i = 0

                # Setup the plot figure with subplots
                fig, ax = plt.subplots(rows, 11 if dataset != "RFW" else 6, figsize=(44 if dataset != "RFW" else 24, 4 * rows))
                plt.title(model + " on " + dataset + " - " + setting)

                # Plot each probe image and its top-k matches
                for k, v in ranks.items():
                    gr_idx = i // n_samples

                    ax[i, 0].axis("off")
                    probe_img = k.replace(path_conv_dict[dataset]["HR"], path_conv_dict[dataset]["LR"]) if setting.startswith("LR") else k
                    probe = cv2.imread(probe_img)
                    ax[i, 0].imshow(probe[:, :, ::-1])
                    ax[i, 0].set_title("Probe Image - " + groups[gr_idx])

                    # Plot the matched images
                    for j in range(10 if dataset != "RFW" else 5):
                        ax[i, j + 1].set_xticks([])
                        ax[i, j + 1].set_yticks([])
                        photopath = random.choice(info_dict[str(v["ranked_classes"][j])][3:])
                        photopath = photopath.replace("CelabA", "CelebA")
                        r = cv2.imread(photopath.replace(path_conv_dict[dataset]["HR"], path_conv_dict[dataset]["LR"]) if setting.endswith("LR") else photopath)

                        ax[i, j + 1].imshow(r[:, :, ::-1])

                        # Highlight correct matches
                        if v["class"] == v["ranked_classes"][j]:
                            plt.setp(ax[i, j + 1].spines.values(), color="green")
                            plt.setp([ax[i, j + 1].get_xticklines(), ax[i, j + 1].get_yticklines()], color="green")
                            for axis in ['top', 'bottom', 'left', 'right']:
                                ax[i, j + 1].spines[axis].set_linewidth(5.5)

                        ax[i, j + 1].set_title("Rank #" + str(j + 1) + " - " + v["groups"][j])
                    i += 1

                plt.savefig(img_path, bbox_inches="tight", dpi=150)
                print(model + " on " + dataset + " - " + setting + ": DONE!")