from pathlib import Path
import shutil
import cv2
import numpy as np
import os
import random
import albumentations as A
from utils import (
    addImageAugmentations,
    generateRealisticBackground,
    getAllFilesRecursively,
    getImageTemplate,
    getAllFolders,
    getOutputSize,
    getRandomString,
    getRotation,
    getScale,
    moveValidationFiles,
    rotateImage,
    skrewImage,
    writeDatasetConfig
)

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

    if os.path.exists(datasetFolder):
        shutil.rmtree(datasetFolder)
    os.makedirs(datasetFolder, exist_ok=False)
    os.makedirs(imagesTrainFolder, exist_ok=False)
    os.makedirs(imagesValFolder, exist_ok=False)
    os.makedirs(labelsTrainFolder, exist_ok=False)
    os.makedirs(labelsValFolder, exist_ok=False)
    os.makedirs(masksTrainFolder, exist_ok=False)
    os.makedirs(masksValFolder, exist_ok=False)

    totalImagesPerClass = 5000
    classNames = [
        "cni-ancien-recto",
        "cni-ancien-verso",
        "cni-nouveau-recto",
        "cni-nouveau-verso",
        "passeport",
        "permis-de-conduire",
        "carte-vitale"
    ]
    allBackgroundPossiblePaths = getAllFilesRecursively(backgroundsPath)

    for j, className in enumerate(classNames):
        backgroundOnlyCount = int(totalImagesPerClass * 0.2)
        orientationSplitCount = int((totalImagesPerClass) * 0.5)

        for i in range(totalImagesPerClass):
            isBackgroundOnly = i < backgroundOnlyCount
            useLandscape = i > orientationSplitCount
            outputWidth, outputHeight = getOutputSize(useLandscape)
            background = generateRealisticBackground(
                outputWidth,
                outputHeight,
                isBackgroundOnly,
                allBackgroundPossiblePaths[random.randint(0, len(allBackgroundPossiblePaths) - 1)] if isBackgroundOnly else None
            )

            if not isBackgroundOnly:
                imageTemplate = getImageTemplate(templatesPath, className, useLandscape)
                imageTemplateHeight, imageTemplateWidth = imageTemplate.shape[:2]
                rotation = getRotation(useLandscape)
                scale = getScale(useLandscape, rotation)

                skrewedImage, skrewMatrix = skrewImage(imageTemplate, imageTemplateWidth, imageTemplateHeight)
                rotatedImage, rotationMatrix, angle = rotateImage(skrewedImage, imageTemplateWidth, imageTemplateHeight, rotation)
                scaleWidth = int(rotatedImage.shape[1] * scale)
                scaleHeight = int(rotatedImage.shape[0] * scale)
                scaledImage = cv2.resize(rotatedImage, (scaleWidth, scaleHeight), interpolation=cv2.INTER_LINEAR)

                alphaChannel = scaledImage[:, :, 3] / 255.0
                mask = (alphaChannel * 255).astype(np.uint8)
                mask_full = np.zeros((outputHeight, outputWidth), dtype=np.uint8)

                scaleWidth = min(scaleWidth, outputWidth)
                scaleHeight = min(scaleHeight, outputHeight)
                maxX = max(0, outputWidth - scaleWidth)
                maxY = max(0, outputHeight - scaleHeight)
                xOffset = random.randint(0, maxX) if maxX > 0 else (outputWidth - scaleWidth) // 2
                yOffset = random.randint(0, maxY) if maxY > 0 else (outputHeight - scaleHeight) // 2

                mask_full[yOffset:yOffset + scaleHeight, xOffset:xOffset + scaleWidth] = mask[:scaleHeight, :scaleWidth]
                for c in range(3):
                    background[yOffset:yOffset + scaleHeight, xOffset:xOffset + scaleWidth, c] = (
                        alphaChannel[:scaleHeight, :scaleWidth] * scaledImage[:scaleHeight, :scaleWidth, c] +
                        (1.0 - alphaChannel[:scaleHeight, :scaleWidth]) * background[yOffset:yOffset + scaleHeight, xOffset:xOffset + scaleWidth, c]
                    )

                background = addImageAugmentations(background)

                xCenter = (xOffset + scaleWidth / 2) / outputWidth
                yCenter = (yOffset + scaleHeight / 2) / outputHeight
                width = scaleWidth / outputWidth
                height = scaleHeight / outputHeight

            fileName = f"{getRandomString()}"
            imageFile = os.path.join(imagesTrainFolder, f"{fileName}.jpg")
            cv2.imwrite(imageFile, background)

            if not isBackgroundOnly:
                labelFile = os.path.join(labelsTrainFolder, f"{fileName}.txt")
                with open(labelFile, "w") as f:
                    contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        if len(contour) >= 3:
                            f.write(f"{j}")
                            for point in contour:
                                x, y = point[0]
                                x = x / outputWidth
                                y = y / outputHeight
                                f.write(f" {x:.6f} {y:.6f}")
                            f.write("\n")

                maskFile = os.path.join(masksTrainFolder, f"{fileName}.jpg")
                cv2.imwrite(maskFile, mask_full)

            print(f"[{i:04d}/{totalImagesPerClass}] '{Path(imageFile).name}'{' (background)' if isBackgroundOnly else ''} created for '{className}'")

    moveValidationFiles(labelsTrainFolder, imagesValFolder, labelsValFolder, masksValFolder, percentage=0.1)
    writeDatasetConfig(datasetFolder, imagesTrainFolder, imagesValFolder, classNames)