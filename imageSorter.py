import os
import glob
import shutil

# ------------------------------------------------------------------------#
#                               Define paths                              #
# ------------------------------------------------------------------------#

# define paths to images and destination folder
source = "archive/chest_xray"
dest = "archive/chest_xray_sorted"
trainDir = "archive/chest_xray/train"
testDir = "archive/chest_xray/test"
valDir = "archive/chest_xray/val"


# ------------------------------------------------------------------------#
#                             Create folders                              #
# ------------------------------------------------------------------------#


# helper function for creating folders and subfolders for sorting images
def createFolders(dest, labels, folders):
    for folder in folders:

        pathMain = os.path.join(dest, folder)
        if not os.path.exists(pathMain):
            os.makedirs(pathMain)

        for label in labels:
            pathSub = os.path.join(pathMain, label)
            if not os.path.exists(pathSub):
                os.makedirs(pathSub)


# ------------------------------------------------------------------------#
#                             Sort images                                 #
# ------------------------------------------------------------------------#


def sortImages(source, dest, folders):

    labelsTemp = ["bacteria", "virus"]
    # print(labelsTemp)

    for folder in folders:
        for label in labelsTemp:
            pathTemp = os.path.join(source, folder, "PNEUMONIA")
            # print(f"pathTemp: {pathTemp}")
            images = glob.glob(
                os.path.join(pathTemp, f"*{label}*.jpeg")
            )  # find all images in designated folder with a specific tag

            for image in images:
                if label == "bacteria":
                    pathDestination = os.path.join(dest, folder, "BACTERIAL")
                else:
                    pathDestination = os.path.join(dest, folder, "VIRAL")

                shutil.move(
                    image, os.path.join(pathDestination, os.path.basename(image))
                )

            # print(f"Number of {label} images found: {len(images)}")
            # print(f"path used: {os.path.join(pathTemp, f"*{label}*.jpeg")}")

    for folder in folders:
        pathTemp = os.path.join(source, folder, "NORMAL")
        pathDestination = os.path.join(dest, folder, "NORMAL")
        shutil.move(pathTemp, pathDestination)


# ------------------------------------------------------------------------#
#                                  Main                                   #
# ------------------------------------------------------------------------#

print("----------------------Starting------------------")
labels = ["NORMAL", "BACTERIAL", "VIRAL"]  # labels to sort images into
folders = ["test", "train", "val"]  # main folders for sorting

createFolders(dest, labels, folders)
sortImages(source, dest, folders)
print("---------------------Done---------------------")
