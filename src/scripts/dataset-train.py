import os
from pathlib import Path
import shutil
from ultralytics import YOLO
from utils import (
    getAllFolders
)

def trainPtModel(baseFolder, datasetFolder, resultsFolder, epochs=50, train=True):
    if train:
        if os.path.exists(resultsFolder):
            shutil.rmtree(resultsFolder)
        model = YOLO("yolo11n-seg.pt")
        model.train(
            data=os.path.join(datasetFolder, "config.yaml"),
            imgsz=320,
            project=baseFolder,
            name=Path(resultsFolder).name,
            epochs=epochs,
            augment=False,
            flipud=0.0,
            fliplr=0.0
        )
    ptModelPath = os.path.join(resultsFolder, "weights", "best.pt")
    print(f"Model '{ptModelPath}' created")
    return ptModelPath

def convertPtModelToTfjsModel(ptModelPath, quantizationEnabled=False):
    command = f"python {os.path.dirname(os.path.abspath(__file__))}/model-convert.py {ptModelPath}"
    if quantizationEnabled:
        command += " --quantizationEnabled"
    os.system(command)

if __name__ == "__main__":
    basePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    (
        datasetFolder,
        imagesTrainFolder,
        imagesValFolder,
        labelsTrainFolder,
        labelsValFolder,
        masksTrainFolder,
        masksValFolder,
        backgroundsPath,
        templatesPath,
        resultsFolder,
        testsFolder,
        testsSamplesFolder,
        testResultsFolder
    ) = getAllFolders(basePath)

    ptModelPath = trainPtModel(basePath, datasetFolder, resultsFolder, epochs=50, train=False)
    convertPtModelToTfjsModel(ptModelPath)
    convertPtModelToTfjsModel(ptModelPath, quantizationEnabled=True)