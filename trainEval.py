import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import loadData
from model import loadModel, saveModel
import time
import curses
from sklearn.metrics import confusion_matrix, classification_report
import os

# ------------------------------------------------------------------------#
#                           Training helper function                      #
# ------------------------------------------------------------------------#


def trainEpoch(model, trainLoader, optimizer, criterionFun, device):
    model.train()  # switch model to training mode
    loss = 0.0  # loss value
    correct = 0  # number of correctly classified images
    total = 0  # total number of images
    accuracy = 0  # accuracy of the model in current epoch

    for image, label in trainLoader:
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()  # reset gradient
        output = model(image)  # prediction output of the model
        tempLoss = criterionFun(output, label)  # calculate the loss
        tempLoss.backward()  # calculate gradient
        optimizer.step()  # update weights

        loss = loss + tempLoss.item()
        _, predicted = torch.max(output, 1)
        # calculate the sum of correctly classified images for accuracy
        correct = correct + (predicted == label).sum().item()
        total = total + label.size(0)  # calculate the total number of images

    accuracy = correct / total * 100  # accuracy in percentage value
    weightedLoss = loss / len(trainLoader)

    return weightedLoss, accuracy


# ------------------------------------------------------------------------#
#                           Evaluate helper function                      #
# ------------------------------------------------------------------------#


def evaluateModel(model, testLoader, criterionFun, device):
    model.eval()
    loss = 0.0
    correct = 0
    total = 0
    accuracy = 0.0
    classAccuracy = 0.0

    classCorrect = torch.zeros(3)
    classTotal = torch.zeros(3)

    className = testLoader.dataset.classes

    with torch.no_grad():  # disable gradient for evaluation
        for image, label in testLoader:
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            lossTemp = criterionFun(output, label)

            loss = loss + lossTemp.item()
            _, predicted = torch.max(output, 1)

            correct = correct + (predicted == label).sum().item()
            total = total + label.size(0)

            for i in range(len(label)):
                labelTemp = label[i].item()

                classCorrect[labelTemp] = (
                    classCorrect[labelTemp] + (predicted[i] == label[i]).item()
                )

                classTotal[labelTemp] += 1

    accuracy = correct / total * 100
    # classAccuracy = classCorrect / classTotal * 100
    weightedLoss = loss / len(testLoader)

    # TEMP - print accuracy per class
    # print("\nAccuracy per class:")
    # for i, name in enumerate(className):
    #    print(f"Accuracy for class {name}: {classAccuracy[i]:.2f}%")

    return weightedLoss, accuracy


# ------------------------------------------------------------------------#
#                                  Read stats                             #
# ------------------------------------------------------------------------#


def readReport(reportPath):
    # default values
    bestAcc = 0.0
    bestLoss = float("inf")
    totalTime = 0.0

    if not os.path.exists(reportPath):
        return bestAcc, bestLoss, totalTime  # return default values
    else:
        with open(reportPath, "r") as f:
            lines = f.readlines()

            for line in lines:
                if "Best accuracy: " in line:
                    try:
                        bestAcc = float(line.split(":")[1].strip())
                    except ValueError:
                        pass  # ignore the line if the value is not correct
                elif "Best loss: " in line:
                    try:
                        bestLoss = float(line.split(":")[1].strip())
                    except ValueError:
                        pass  # ignore the line if the value is not correct
                elif "Total training time: " in line:
                    try:
                        # totalTime = parseTime(line.split(":", 1)[1].strip())
                        totalTime = float(line.split(":")[1].strip())
                    except ValueError:
                        pass  # ignore the line if the value is not correct

        return bestAcc, bestLoss, totalTime


# ------------------------------------------------------------------------#
#                                 Format time                             #
# ------------------------------------------------------------------------#

# FIX READING AND WRITTING FORMATED TIME!!!!


def formatTime(Tseconds):
    hours = int(Tseconds // 3600)
    minutes = int((Tseconds - hours * 3600) // 60)
    seconds = int(Tseconds - hours * 3600 - minutes * 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# ------------------------------------------------------------------------#
#                               Unformat time                             #
# ------------------------------------------------------------------------#


def parseTime(timeStr):
    try:
        hours, minutes, seconds = map(int, timeStr.split(":"))
        return hours * 3600 + minutes * 60 + seconds
    except ValueError:
        return 0.0


# ------------------------------------------------------------------------#
#                              Main training loop                         #
# ------------------------------------------------------------------------#


# main training function
def trainModel(
    stdscr,
    dataPath="archive/chest_xray_sorted",
    modelPath="models/KlemenDela.pth",
    reportPath="models/reports/KlemenDela.txt",
    batchSize=32,
    epochs=10,
    learnRate=0.01,
):
    titleSize = 80
    titleSizeTemp = int((titleSize - 14) / 2)
    titleSizeTemp2 = int((titleSize - 8) / 2)
    stdscr.addstr(
        4, 0, f"<" + titleSizeTemp * "-" + "Training  Mode" + titleSizeTemp * "-" + ">"
    )

    # cuda check
    stdscr.addstr(5, 0, ">CUDA check: ")
    stdscr.refresh()
    if torch.cuda.is_available():
        stdscr.addstr(5, len(">CUDA check: "), "Cuda cores are available")
        device = torch.device("cuda")
    else:
        stdscr.addstr(
            5,
            len(">CUDA check: "),
            "Cuda cores aren't available, switching to CPU mode",
        )
        device = torch.device("cpu")
    stdscr.refresh()

    # load model
    model = loadModel(loadPath=modelPath, classN=3)

    # load data
    trainLoader, testLoader, valLoader = loadData(
        dataPath=dataPath, batchSize=batchSize
    )

    # define optimizer
    # optimizer = optim.SGD(model.parameters(), lr=learnRate)
    optimizer = optim.Adam(model.parameters(), lr=learnRate)  # testing new optimizer
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.5
    )  # testing scheduler
    criterionFun = nn.CrossEntropyLoss()

    # print UI shell
    stdscr.addstr(7, 0, f"Epoch [0/{epochs}]")
    stdscr.addstr(8, 0, f"Train loss:  /  | Train accuracy:  / ")
    stdscr.addstr(9, 0, f"Test loss:  /  | Test accuracy:  / ")
    stdscr.addstr(10, 0, f"Progress: [{50 * '.'}] 0/{epochs}")
    stdscr.refresh()

    # read report
    bestAcc, bestLoss, totalTime = readReport(reportPath.replace(".pth", "_report.txt"))

    # start timer
    startTime = time.time()

    for epoch in range(epochs):
        trainLoss, trainAcc = trainEpoch(
            model=model,
            trainLoader=trainLoader,
            optimizer=optimizer,
            criterionFun=criterionFun,
            device=device,
        )

        testLoss, testAcc = evaluateModel(
            model=model, testLoader=testLoader, criterionFun=criterionFun, device=device
        )

        # print out training and testing stats for current epoch
        stdscr.addstr(7, 0, f"Epoch [{epoch+1}/{epochs}]")  # print current epoch
        stdscr.addstr(
            8, 0, f"Train loss: {trainLoss:.2f} | Train accuracy: {trainAcc:.2f}%"
        )
        stdscr.addstr(
            9, 0, f"Test loss: {testLoss:.2f} | Test accuracy: {testAcc:.2f}%"
        )

        # progress bar
        progress = int((epoch + 1) / epochs * 50)  # length of progress bar: 50
        stdscr.addstr(
            10,
            0,
            f"Progress: [{progress * '|'}{(50 - progress) * '.'}] {epoch+1}/{epochs}",
        )

        stdscr.move(12, 0)
        stdscr.clrtoeol()
        stdscr.refresh()
        time.sleep(1)

        # if there is new best model save it
        if testAcc > bestAcc:
            # save current best values and model
            bestAcc = testAcc
            bestLoss = testLoss
            bestEpoch = epoch + 1
            stdscr.addstr(12, 0, "New best model found!, saving model...")
            stdscr.refresh()
            saveModel(model=model, savePath=modelPath)

        scheduler.step()
    # stop timer
    finishTime = time.time()
    currentTime = finishTime - startTime  # total time spent
    totalTime = totalTime + currentTime

    # final outputs
    # stdscr.addstr(14, 0, f"Time: {formatTime(currentTime)}")
    stdscr.addstr(14, 0, f"Time: {currentTime:.2f}s")
    # stdscr.addstr(15, 0, "Training finished, press any button to continue...")
    stdscr.addstr(
        4, 0, f"<" + titleSizeTemp2 * "-" + "Finished" + titleSizeTemp2 * "-" + ">"
    )
    stdscr.addstr(16, 0, "Saving model summary...")
    stdscr.refresh()
    # stdscr.getch()

    # model summary
    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in testLoader:

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    cMatrix = confusion_matrix(y_true, y_pred)
    modelReport = classification_report(
        y_true, y_pred, target_names=["NORMAL", "BACTERIAL", "VIRAL"]
    )

    # save path check
    if not os.path.exists("models/reports"):
        os.makedirs("models/reports")

    # save model summary
    summaryPath = reportPath.replace(".pth", "_report.txt")
    with open(summaryPath, "w") as f:
        # basic stats
        f.write(f"Model summary for: {modelPath}\n\n")
        f.write(f"Best accuracy: {bestAcc:.2f}\n")
        f.write(f"Best loss: {bestLoss:.2f}\n")
        f.write(f"Total training time: {totalTime:.2f}\n\n")

        # confusion matrix
        f.write(f"Confusion matrix:\n{cMatrix}\n\n")

        # classification report
        f.write(f"Classification report:\n{modelReport}\n")

    # next step
    items = ["<Go to summary>", "<Back>"]
    selection = 0

    while True:
        for id, option in enumerate(items):
            if id == selection:
                stdscr.addstr(id + 18, 0, option, curses.color_pair(1))
            else:
                stdscr.addstr(id + 18, 0, option)

        stdscr.refresh()

        pressedKey = stdscr.getch()

        if pressedKey == curses.KEY_UP and selection > 0:
            selection = selection - 1
        elif pressedKey == curses.KEY_DOWN and selection < 1:
            selection = selection + 1
        elif pressedKey == 10:  # enter key
            return selection


# ------------------------------------------------------------------------#
#                               Code test                                 #
# ------------------------------------------------------------------------#

if __name__ == "__main__":
    trainModel()
