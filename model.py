import torch
import torch.nn as nn
import os
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ------------------------------------------------------------------------#
#                       Create EfficientNet model                         #
# ------------------------------------------------------------------------#


class NeuralModel(nn.Module):
    def __init__(self, classNum=3):
        super().__init__()

        # load pretrained model
        weights = EfficientNet_B0_Weights.DEFAULT
        self.model = efficientnet_b0(weights=weights)

        # disable feature extraction layers
        for param in list(self.model.features.parameters())[:5]:
            param.requires_grad = False

        # extract dimensions of the last layer
        features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(  # add new layers in sequential order
            # -----------------------------------
            # Original layer setup
            # nn.Linear -> applies linear affine transformation -> TLDR: linear image scalling
            # nn.Linear(features, 128),  # 1280 -> 128
            # nn.Relu() -> activation function which gives non linearty factor to neural network
            # nn.ReLU(),
            ## nn.Dropout() -> randomly excludes certain neurons to prevent overfitting
            # nn.Dropout(0.5),
            # nn.Linear(128, classNum),  # 128 -> 3
            # ------------------------------------
            # potential layer setup - TEST IT!
            # nn.Linear(features, 512),
            # nn.ReLU(),
            # nn.Linear(512, 128),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(128, classNum),
            # ---------------------------------
            # Final layer setup
            nn.Linear(features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),  # try 0.2-0.3
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, classNum),
        )

    def forward(self, x):  # feed forward
        return self.model(x)


# ------------------------------------------------------------------------#
#                              Save model                                 #
# ------------------------------------------------------------------------#


def saveModel(model, savePath="models/efficientNet.pth"):

    if not os.path.exists("models"):  # check for "models" folder
        os.makedirs("models")

    torch.save(model.state_dict(), savePath)  # save model to "savePath"
    # print(f"Model was saved to: {savePath}")


# ------------------------------------------------------------------------#
#                              Load model                                 #
# ------------------------------------------------------------------------#


def loadModel(loadPath="models/efficientNet.pth", classN=3):
    tempModel = NeuralModel(classNum=classN)

    tempModel.load_state_dict(torch.load(loadPath, map_location=torch.device("cpu")))

    if torch.cuda.is_available():
        tempModel.to(torch.device("cuda"))
    else:
        tempModel.to(torch.device("cpu"))

    # print(f"Model was loaded from: {loadPath}")
    return tempModel


# ------------------------------------------------------------------------#
#                               Code test                                 #
# ------------------------------------------------------------------------#


if __name__ == "__main__":
    modelTest = NeuralModel(classNum=3)
    saveModel(model=modelTest)

    modelTest = loadModel()
    # print("\n", modelTest)
