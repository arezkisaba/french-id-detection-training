import os
import shutil
import sys
from ultralytics import YOLO
from utils import (
    getAllFolders
)

def convertPtModelToTfjsModel(ptModelPath, quantizationEnabled=True):
    ptBasePath = os.path.dirname(ptModelPath)
    outputDirForWeb = os.path.join(ptBasePath, "best_web_model")
    finalOutputDirForWeb = outputDirForWeb + ("_quantized" if quantizationEnabled else "_original")
    if os.path.isdir(finalOutputDirForWeb):
        shutil.rmtree(finalOutputDirForWeb)

    model = YOLO(ptModelPath)
    model.export(
        format="tfjs",
        imgsz=320,
        opset=12,
        half=False,
        simplify=False,
        int8=quantizationEnabled
    )
    
    shutil.rmtree(os.path.join(ptBasePath, "best_saved_model"))
    os.remove(os.path.join(ptBasePath, "best.pb"))
    os.remove(os.path.join(ptBasePath, "best.onnx"))
    os.rename(outputDirForWeb, finalOutputDirForWeb)

    if quantizationEnabled:
        print(f"Model 'tfjs' created with 8-bit quantization")
    else:
        print(f"Model 'tfjs' created")

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

    ptModelPath = sys.argv[1]
    quantizationEnabled = '--quantizationEnabled' in sys.argv
    convertPtModelToTfjsModel(ptModelPath, quantizationEnabled)