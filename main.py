# ------------------------------------------------------------------------#
#                                 Imports                                 #
# ------------------------------------------------------------------------#

import os
from trainEval import trainModel
import time
from model import NeuralModel, saveModel, loadModel
import curses
from PIL import Image
from torchvision import transforms
import torch
import subprocess
import sys
import os


# ------------------------------------------------------------------------#
#                              Clear screen                               #
# ------------------------------------------------------------------------#


def clearScreen(stdscr, title=False):
    stdscr.clear()

    if title:
        printTitle(stdscr)


# ------------------------------------------------------------------------#
#                              Print title                                #
# ------------------------------------------------------------------------#


def printTitle(stdscr):
    titleSize = 80
    titleSizeTemp = int((titleSize - 26) / 2)

    stdscr.addstr(0, 0, "#" + "-" * titleSize + "#")
    stdscr.addstr(
        1,
        0,
        "#"
        + " " * titleSizeTemp
        + "Chest X-ray Classification"
        + " " * titleSizeTemp
        + "#",
    )
    stdscr.addstr(2, 0, "#" + "-" * titleSize + "#")
    stdscr.refresh()


# ------------------------------------------------------------------------#
#                              Main menu                                  #
# ------------------------------------------------------------------------#


def mainMenu(stdscr):
    clearScreen(stdscr, True)

    items = [
        "Create new model",
        "Train model",
        "Test model",
        "Delete model",
        "Summary",
        "Exit",
    ]
    selection = 0

    while True:
        clearScreen(stdscr, True)
        for id, option in enumerate(items):
            if id == selection:
                stdscr.addstr(id + 4, 1, f">{option}", curses.color_pair(1))
            else:
                stdscr.addstr(id + 4, 3, option)

        stdscr.addstr(
            30,
            0,
            "[Enter] Select  |  [↑↓] Navigate  ",
        )

        stdscr.refresh()

        pressedKey = stdscr.getch()

        if pressedKey == curses.KEY_UP and selection > 0:  # move up - arrow up
            selection = selection - 1
        elif pressedKey == curses.KEY_DOWN and selection < 5:  # move down - arrow down
            selection = selection + 1
        elif pressedKey == 10:  # enter key
            return selection  # return index of selected option


# ------------------------------------------------------------------------#
#                         Submenu - Create model                          #
# ------------------------------------------------------------------------#


def submenuCreate(stdscr):

    clearScreen(stdscr, True)
    stdscr.addstr(4, 0, "Enter model name: ")
    stdscr.refresh()
    curses.echo()  # enable text input
    curses.curs_set(1)  # enable cursor
    modelName = stdscr.getstr(4, len("Enter model name: "), 20).decode("utf-8")
    curses.curs_set(0)  # disable cursor
    curses.noecho()  # disable text input

    model = NeuralModel(classNum=3)
    saveModel(model, f"models/{modelName}.pth")
    time.sleep(1)
    stdscr.addstr(
        6, 0, f"Model {modelName} was created and saved to: models/{modelName}.pth"
    )
    stdscr.addstr(30, 0, "Press any key to return")
    stdscr.refresh()
    stdscr.getch()


# ------------------------------------------------------------------------#
#                        Submenu - Display models                         #
# ------------------------------------------------------------------------#


def listModels(stdscr):

    modelList = []

    if not os.path.exists("models"):  # check for "models" folder
        clearScreen(stdscr, True)
        stdscr.addstr(4, 0, "Error: folder 'models' doesn't exist")
        time.sleep(2)
        return modelList

    if len(os.listdir("models")) == 0:  # check if folder is empty
        clearScreen(stdscr, True)
        stdscr.addstr(4, 0, "Folder 'models' is empty")
        stdscr.addstr(5, 0, "Create new model first")
        time.sleep(2)
        return modelList

    for model in os.listdir("models"):
        if model.endswith(".pth"):
            modelList.append(model)

    return modelList


# ------------------------------------------------------------------------#
#                          Submenu - Select model                         #
# ------------------------------------------------------------------------#


def selectModel(stdscr, flag=0):
    # flag: "0" -> "Select model", "1" -> "Select model to train", "2" -> "Select model to delete", "3" -> "Select model to test"

    curses.curs_set(0)
    stdscr.refresh()

    modelList = listModels(stdscr)

    if len(modelList) != 0:

        selection = 0

        while True:
            clearScreen(stdscr, True)
            for id, model in enumerate(modelList):
                if id == selection:
                    stdscr.addstr(id + 6, 1, f">{model}", curses.color_pair(1))
                else:
                    stdscr.addstr(id + 6, 3, model)

                stdscr.addstr(
                    30,
                    0,
                    "[Enter] Select Model  |  [↑↓] Navigate  |  [Backspace] Return",
                )

            match flag:
                case 0:
                    stdscr.addstr(4, 0, "Select model:")
                case 1:
                    stdscr.addstr(4, 0, "Select model to train:")
                case 2:
                    stdscr.addstr(4, 0, "Select model to delete:")
                case 3:
                    stdscr.addstr(4, 0, "Select model to test:")

            stdscr.refresh()
            key = stdscr.getch()

            if key == curses.KEY_UP and selection > 0:  # move up
                selection = selection - 1
            elif key == curses.KEY_DOWN and selection < len(modelList) - 1:  # move down
                selection = selection + 1
            elif key == 8:
                return None
            elif key == 10:  # enter key
                return modelList[selection]


# ------------------------------------------------------------------------#
#                          Submenu - Train model                          #
# ------------------------------------------------------------------------#


def submenuTrain(stdscr):
    clearScreen(stdscr, True)
    stdscr.addstr(4, 0, "Select model to train:")
    stdscr.refresh()

    selectedModel = selectModel(stdscr, flag=1)
    if selectedModel is not None:
        batchSize, epochs, learnRate = getParameters(stdscr)
        clearScreen(stdscr, True)
        selection = trainModel(
            stdscr,
            modelPath=f"models/{selectedModel}",
            reportPath=f"models/reports/{selectedModel}",
            batchSize=int(batchSize),
            epochs=int(epochs),
            learnRate=float(learnRate),
        )

        if selection == 0:
            submenuSum(
                stdscr,
                reportPath=f"models/reports/{selectedModel}".replace(
                    ".pth", "_report.txt"
                ),
                flag=0,
            )


# ------------------------------------------------------------------------#
#                           Submenu - Summary                             #
# ------------------------------------------------------------------------#


def submenuSum(stdscr, reportPath, flag=0):
    clearScreen(stdscr, True)

    # page title
    titleSize = 80
    titleSizeTemp = int((titleSize - 14) / 2)
    stdscr.addstr(
        4, 0, f"<" + titleSizeTemp * "-" + "Model  Summary" + titleSizeTemp * "-" + ">"
    )

    if os.path.exists(reportPath):
        with open(reportPath, "r") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            stdscr.addstr(i + 6, 0, line.rstrip())

        stdscr.addstr(30, 0, "Press any key to return...")
        stdscr.refresh()
        stdscr.getch()
        if flag == 1:
            submenuSumTemp(stdscr)
    else:
        stdscr.addstr(6, 0, f"Error, {reportPath} doesn't exists!")
        stdscr.addstr(30, 0, "Press any key to return...")
        stdscr.refresh()
        stdscr.getch()

        if flag == 1:
            submenuSumTemp(stdscr)


# ------------------------------------------------------------------------#
#                         Submenu - Summary Temp                          #
# ------------------------------------------------------------------------#


def submenuSumTemp(stdscr):
    clearScreen(stdscr, True)
    stdscr.addstr(4, 0, "Select model:")
    selection = selectModel(stdscr, flag=0)

    if selection is not None:
        submenuSum(
            stdscr,
            reportPath=f"models/reports/{selection}".replace(".pth", "_report.txt"),
            flag=1,
        )


# ------------------------------------------------------------------------#
#                       Submenu - Get parameters                          #
# ------------------------------------------------------------------------#


def getParameters(stdscr):
    clearScreen(stdscr, True)
    stdscr.addstr(4, 0, "Enter training parameters:")

    # get batch size
    stdscr.addstr(6, 0, "Batch size: ")
    stdscr.refresh()
    curses.echo()
    curses.curs_set(1)
    batchSize = stdscr.getstr(6, len("Batch size: "), 20).decode("utf-8")

    while not batchSize.isdigit():  # check if input is a number
        stdscr.move(6, 0)
        stdscr.clrtoeol()
        stdscr.addstr(8, 0, "<Error: batch size must be a number!>")
        stdscr.addstr(6, 0, "Batch size: ")
        stdscr.refresh()
        curses.echo()
        curses.curs_set(1)
        batchSize = stdscr.getstr(6, len("Batch size: "), 20).decode("utf-8")

    stdscr.move(8, 0)
    stdscr.clrtoeol()
    curses.curs_set(0)
    curses.noecho()

    # get epochs
    stdscr.addstr(8, 0, "Epochs: ")
    stdscr.refresh()
    curses.echo()
    curses.curs_set(1)
    epochs = stdscr.getstr(8, len("Epochs: "), 20).decode("utf-8")

    while not epochs.isdigit():  # check if input is a number
        stdscr.move(8, 0)
        stdscr.clrtoeol()
        stdscr.addstr(10, 0, "<Error: epochs must be a number!>")
        stdscr.addstr(8, 0, "Epochs: ")
        stdscr.refresh()
        curses.echo()
        curses.curs_set(1)
        epochs = stdscr.getstr(8, len("Epochs: "), 20).decode("utf-8")

    stdscr.move(10, 0)
    stdscr.clrtoeol()
    curses.curs_set(0)
    curses.noecho()

    # get learning rate
    stdscr.addstr(10, 0, "Learning rate: ")
    stdscr.refresh()
    curses.echo()
    curses.curs_set(1)
    learnRate = stdscr.getstr(10, len("Learning rate: "), 20).decode("utf-8")

    while not learnRate.replace(".", "", 1).isdigit():  # check if input is a number
        stdscr.move(10, 0)
        stdscr.clrtoeol()
        stdscr.addstr(12, 0, "<Error: learning rate must be a number!>")
        stdscr.addstr(10, 0, "Learning rate: ")
        stdscr.refresh()
        curses.echo()
        curses.curs_set(1)
        learnRate = stdscr.getstr(10, len("Learning rate: "), 20).decode("utf-8")

    stdscr.move(12, 0)
    stdscr.clrtoeol()
    curses.curs_set(0)
    curses.noecho()

    return batchSize, epochs, learnRate


# ------------------------------------------------------------------------#
#                          Submenu - Delete model                         #
# ------------------------------------------------------------------------#


def submenuDelete(stdscr):
    clearScreen(stdscr, True)

    stdscr.addstr(4, 0, "Select model to delete:")
    stdscr.refresh()

    selectedModel = selectModel(stdscr, flag=2)
    if selectedModel is not None:
        makeYourChoice(stdscr, selectedModel)
        submenuDelete(stdscr)


# ------------------------------------------------------------------------#
#                          Submenu - Delete model                         #
# ------------------------------------------------------------------------#


def makeYourChoice(stdscr, selectedModel):

    choices = ["Yes", "No"]
    selection = 0

    while True:
        clearScreen(stdscr, True)

        stdscr.addstr(4, 0, f"Are you sure you want to delete {selectedModel}?")

        for id, choice in enumerate(choices):
            if id == selection:
                stdscr.addstr(id + 6, 1, f">{choice}", curses.color_pair(1))
            else:
                stdscr.addstr(id + 6, 3, choice)

        stdscr.refresh()
        key = stdscr.getch()

        if key == curses.KEY_UP and selection > 0:
            selection = selection - 1
        elif key == curses.KEY_DOWN and selection < 1:
            selection = selection + 1
        elif key == 10:
            if selection == 0:  # if yes is selected
                # remove model file
                modelPath = f"models/{selectedModel}"
                os.remove(modelPath)

                # remove report file
                reportPath = f"models/reports/{selectedModel}".replace(
                    ".pth", "_report.txt"
                )
                if os.path.exists(reportPath):
                    os.remove(reportPath)

                clearScreen(stdscr, True)
                stdscr.addstr(4, 0, f"Model {selectedModel} was deleted")
                stdscr.addstr(30, 0, "Press any key to return")
                stdscr.refresh()
                stdscr.getch()
                break
            else:
                break


# ------------------------------------------------------------------------#
#                           Submenu - Test model                          #
# ------------------------------------------------------------------------#


def submenuTest(stdscr):
    clearScreen(stdscr, True)
    stdscr.addstr(4, 0, "Select model to test:")
    selection = selectModel(stdscr, flag=3)

    clearScreen(stdscr, True)
    titleSize = 80
    titleSizeTemp = int((titleSize - 10) / 2)
    stdscr.addstr(
        4, 0, f"<" + titleSizeTemp * "-" + "Test Model" + titleSizeTemp * "-" + ">"
    )

    if selection is not None:
        modelPath = f"models/{selection}"

        if os.path.exists(modelPath):
            testModel(stdscr, modelPath=modelPath, selectedModel=selection)
            submenuTest(stdscr)
        else:
            stdscr.addstr(6, 0, f"Error, selected model doesn't exist!")
            stdscr.refresh()
            stdscr.getch()
            return


def loadTestImages(valPath):

    images = []  # array for test images

    for root, _, files in os.walk(valPath):
        for file in files:
            if file.endswith(".jpeg"):
                images.append(os.path.join(root, file))

    return images


def testModel(
    stdscr, modelPath, selectedModel, valPath="archive/chest_xray_sorted/val"
):
    images = loadTestImages(valPath)
    titleSize = 80
    titleSizeTemp = int((titleSize - 10) / 2)

    if not images:
        stdscr.addstr(8, 0, f"No valid images were found!")
        stdscr.addstr(10, 0, "Press any key to return...")
        stdscr.refresh()
        stdscr.getch()
        return

    selection = 0  # index of selected image

    while True:

        clearScreen(stdscr, True)
        stdscr.addstr(
            4, 0, f"<" + titleSizeTemp * "-" + "Test Model" + titleSizeTemp * "-" + ">"
        )
        stdscr.addstr(6, 0, f"Selected model: {selectedModel}")
        stdscr.addstr(8, 0, "Select image to classify:")

        for i, image in enumerate(images):
            if i == selection:
                stdscr.addstr(i + 9, 1, f">{os.path.basename(image)}", curses.A_REVERSE)
            else:
                stdscr.addstr(i + 9, 3, os.path.basename(image))

        stdscr.addstr(
            30,
            0,
            "[Enter] Select Image  |  [V] View Image  |  [↑↓] Navigate  |  [Backspace] Return",
        )
        stdscr.refresh()
        key = stdscr.getch()

        if key == curses.KEY_UP and selection > 0:
            selection = selection - 1
        elif key == curses.KEY_DOWN and selection < len(images) - 1:
            selection = selection + 1
        elif key == 118:
            Image.open(images[selection]).show()
        elif key == 8:
            return
        elif key == 10:
            selectedImage = images[selection]
            break

    prediction = imagePrediction(modelPath, selectedImage)

    # print out the prediction
    clearScreen(stdscr, True)
    stdscr.addstr(
        4, 0, f"<" + titleSizeTemp * "-" + "Test Model" + titleSizeTemp * "-" + ">"
    )
    stdscr.addstr(6, 0, f"Selected model: {selectedModel}")
    stdscr.addstr(8, 0, f"Selected image: {os.path.basename(selectedImage)}")
    stdscr.addstr(9, 0, f"Model prediction: {prediction}")
    stdscr.addstr(11, 0, "Press any key to return...")
    stdscr.refresh()
    stdscr.getch()
    return


def imagePrediction(modelPath, imagePath):

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    image = Image.open(imagePath).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    model = loadModel(modelPath, classN=3)
    model.eval()

    with torch.no_grad():
        output = model(image)
        _, prediction = torch.max(output, 1)

    labels = ["BACTERIAL", "NORMAL", "VIRAL"]

    return labels[prediction.item()]


# ------------------------------------------------------------------------#
#                               Main code                                 #
# ------------------------------------------------------------------------#


def main(stdscr):
    curses.curs_set(0)
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)

    while True:
        selection = mainMenu(stdscr)

        match selection:
            case 0:
                submenuCreate(stdscr)
            case 1:
                submenuTrain(stdscr)
            case 2:
                submenuTest(stdscr)
            case 3:
                submenuDelete(stdscr)
            case 4:
                submenuSumTemp(stdscr)

            case 5:
                break


if __name__ == "__main__":
    if os.environ.get("RUNNING_IN_NEW_POWERSHELL") != "1":
        script_path = os.path.abspath(sys.argv[0])

        powershell_command = (
            f"$Env:RUNNING_IN_NEW_POWERSHELL=1; "
            f"$Host.UI.RawUI.BufferSize = New-Object Management.Automation.Host.Size(120,60); "
            f"$Host.UI.RawUI.WindowSize = New-Object Management.Automation.Host.Size(120,60); "
            f'python "{script_path}"; '
            'Read-Host "Press Enter to exit..."'
        )

        subprocess.Popen(
            [
                "cmd.exe",
                "/c",
                "start",
                "powershell.exe",
                "-NoExit",
                "-Command",
                powershell_command,
            ]
        )

        sys.exit(0)

    import curses

    curses.wrapper(main)
