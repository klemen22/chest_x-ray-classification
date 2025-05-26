import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt

dataPath = "archive/chest_xray_sorted"

batchSize = 32  # batch size depends on the GPU memory
imageSize = (224, 224)  # new size of the image

# ------------------------------------------------------------------------#
#                           Helper functions                              #
# ------------------------------------------------------------------------#

transform = transforms.Compose(
    [
        # --------------------------------------
        # Original parameters
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        # --------------------------------------
        # Final parameters
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)


def loadData(dataPath, batchSize):
    # define paths to the training, testing and validation data
    trainPath = os.path.join(dataPath, "train")
    testPath = os.path.join(dataPath, "test")
    valPath = os.path.join(dataPath, "val")
    VIRALpath = os.path.join(valPath, "VIRAL")

    if os.path.exists(VIRALpath):
        print("Deleting VIRAL folder")
        os.rmdir(VIRALpath)

    trainData = datasets.ImageFolder(root=trainPath, transform=transform)
    testData = datasets.ImageFolder(root=testPath, transform=transform)
    valData = datasets.ImageFolder(root=valPath, transform=transform)

    # added weighted random sampler for testing
    labels = [label for _, label in trainData.samples]
    classCount = np.bincount(labels)
    classWeight = 1.0 / torch.tensor(classCount, dtype=torch.float)
    sampleWeight = [classWeight[label] for label in labels]

    sampler = WeightedRandomSampler(
        sampleWeight, num_samples=len(sampleWeight), replacement=True
    )

    trainLoader = torch.utils.data.DataLoader(
        trainData, batch_size=batchSize, sampler=sampler, num_workers=4
    )
    # put shuffle to false in "testLoader" to fix accuracy in classification_report() function
    testLoader = torch.utils.data.DataLoader(
        testData, batch_size=batchSize, shuffle=False, num_workers=4
    )
    valLoader = torch.utils.data.DataLoader(
        valData, batch_size=batchSize, shuffle=True, num_workers=4
    )
    print(f"Data was loaded from: {dataPath}")
    return trainLoader, testLoader, valLoader


def displayImage(image):
    # helper function to display an image for debugging
    image = image / 2 + 0.5  # unnormalize
    npimg = image.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    trainLoader, testLoader, valLoader = loadData(dataPath, batchSize)

    # ------------------------------------------------------------------------#
    #                                Debugging                                #
    # ------------------------------------------------------------------------#

    # debug prints
    print(f"Classes in train data: {trainLoader.dataset.classes}")
    print(f"Number of batches in train data: {len(trainLoader)}")
    print(f"Number of batches in test data: {len(testLoader)}")
    print(f"Number of batches in validation data: {len(valLoader)}")

    # batch test print
    batch = next(iter(trainLoader))
    images, labels = batch
    # Batch shape: torch.Size([32, 3, 224, 224]), 32 -> bacth size, 3 -> RGB, 224x224 -> image size
    print(f"Batch shape: {images.shape}")
    displayImage(images[0])  # display first image in the batch
