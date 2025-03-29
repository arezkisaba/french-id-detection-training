import os
import cv2
import shutil
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from dataclasses import dataclass
import matplotlib.pyplot as plt
from utils import (
    getAllFilesRecursively,
    getAllFolders
)

def purgeTestOutputFolder(testOutputFolder):
    if os.path.isdir(testOutputFolder):
        shutil.rmtree(testOutputFolder)
    os.makedirs(testOutputFolder)

@dataclass
class SegmentationResult:
    bbox: dict
    maskData: dict
    originalMask: np.ndarray
    
    @classmethod
    def fromYoloResult(cls, result, imageShape):
        if result.masks is None or len(result.masks) == 0:
            return None
        mask = result.masks[0].data[0].cpu().numpy()
        confidence = float(result.boxes[0].conf[0].cpu().numpy())
        bbox = getBoundingBoxMeasurements(result, confidence)
        maskData = getMaskMeasurements(mask, imageShape)
        return cls(bbox=bbox, maskData=maskData, originalMask=mask)

    def drawOverlay(self, image):
        overlayImage = image.copy()
        fillColor = (0, 255, 0)
        borderColor = (0, 150, 0)
        textColor = (255, 255, 255)
        borderThickness = 8
        fontScale = 1.8
        
        maskOverlay = overlayImage.copy()
        cv2.fillPoly(maskOverlay, [self.maskData['box']], fillColor)
        cv2.addWeighted(maskOverlay, 0.3, overlayImage, 0.7, 0, overlayImage)
        
        x, y = int(self.bbox['x']), int(self.bbox['y'])
        w, h = int(self.bbox['width']), int(self.bbox['height'])
        cv2.rectangle(overlayImage, (x, y), (x + w, y + h), borderColor, borderThickness)
        
        text = f"{self.bbox.get('confidence', 0):.2f} {self.maskData['angle']:.1f}deg"
        textSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale, borderThickness)[0]
        
        padding = 20
        bgColor = (0, 100, 0)
        cv2.rectangle(
            overlayImage, 
            (x, y - textSize[1] - padding), 
            (x + textSize[0] + padding, y), 
            bgColor, 
            -1
        )
        
        cv2.putText(
            overlayImage, 
            text,
            (x + padding//2, y - padding//2),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            textColor,
            borderThickness
        )
        
        return overlayImage
    
    def crop(self, image):
        rect = self.maskData['rect']
        box = self.maskData['box']
        width = int(rect[1][0])
        height = int(rect[1][1])
        srcPoints = box.astype("float32")
        dstPoints = np.array([[0, height], [0, 0], [width, 0], [width, height]], dtype="float32")
        M = cv2.getPerspectiveTransform(srcPoints, dstPoints)
        warped = cv2.warpPerspective(image, M, (width, height), borderValue=(255, 255, 255))
        
        if warped.shape[0] > warped.shape[1]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        return warped
    
    def __str__(self):
        return (
            f"Detection: [x:{self.bbox['x']:.1f}, y:{self.bbox['y']:.1f}, "
            f"w:{self.bbox['width']:.1f}, h:{self.bbox['height']:.1f}, "
            f"a:{self.maskData['angle']:.1f}°]"
        )

def getBoundingBoxMeasurements(result, confidence=None):
    if result.boxes is not None and len(result.boxes) > 0:
        box = result.boxes[0].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(float, box)
        width = x2 - x1
        height = y2 - y1
        return {
            'x': x1,
            'y': y1,
            'width': width,
            'height': height,
            'confidence': confidence
        }
    return None

def getMaskMeasurements(mask, imageShape):
    maskResized = cv2.resize(mask, (imageShape[1], imageShape[0]))
    maskBinary = (maskResized > 0.5).astype(np.uint8)
    yCoords, xCoords = np.nonzero(maskBinary)
    points = np.column_stack((xCoords, yCoords))
    
    rect = cv2.minAreaRect(points)
    angle = rect[-1]

    if rect[1][0] < rect[1][1]:
        angle -= 90
    
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    
    return {
        'rect': rect,
        'box': box,
        'angle': angle
    }

def processPrediction(prediction, image, inputImagePath, outputPath, i):
    segmentationResult = SegmentationResult.fromYoloResult(prediction, image.shape)
    if segmentationResult is None:
        return
    
    detectionFileName =  Path(inputImagePath).name.replace(".jpg", f"-det-{i}.jpg")
    cv2.imwrite(os.path.join(outputPath, detectionFileName), segmentationResult.drawOverlay(image))
    rotatedFileName =  Path(inputImagePath).name.replace(".jpg", f"-crop-{i}.jpg")
    cv2.imwrite(os.path.join(outputPath, rotatedFileName), segmentationResult.crop(image))

def processImage(model, inputImagePath, outputPath):
    opencvInputImage = cv2.imread(inputImagePath)
    predictions = model.predict(opencvInputImage)

    i = 1
    for prediction in predictions:
        processPrediction(prediction, opencvInputImage, inputImagePath, outputPath, i)
        i += 1

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

    purgeTestOutputFolder(testResultsFolder)
    model = YOLO(os.path.join(resultsFolder, 'weights/best.pt'))
    files = getAllFilesRecursively(testsSamplesFolder)
    for file in files:
        processImage(model, file, testResultsFolder)
