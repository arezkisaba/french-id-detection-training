import math
from pathlib import Path
import shutil
import cv2
import numpy as np
import os
import random
import string
import yaml
import albumentations as A

def getAllFolders(baseRelativeFolder):
    baseFolder = os.path.abspath(baseRelativeFolder)
    datasetFolder = os.path.abspath(os.path.join(baseFolder, "dataset"))
    imagesTrainFolder = os.path.join(datasetFolder, "images/train")
    imagesValFolder = os.path.join(datasetFolder, "images/val")
    labelsTrainFolder = os.path.join(datasetFolder, "labels/train")
    labelsValFolder = os.path.join(datasetFolder, "labels/val")
    masksTrainFolder = os.path.join(datasetFolder, "masks/train")
    masksValFolder = os.path.join(datasetFolder, "masks/val")
    backgroundsPath = os.path.join(baseRelativeFolder, "openimages", "backgrounds")
    templatesPath = os.path.join(baseRelativeFolder, "templates")
    resultsFolder = os.path.abspath(os.path.join(baseFolder, "results"))
    testsFolder = os.path.abspath(os.path.join(baseFolder, "tests"))
    testsSamplesFolder = os.path.abspath(os.path.join(testsFolder, "samples"))
    testsResultsFolder = os.path.abspath(os.path.join(testsFolder, "results"))
    return (
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
        testsResultsFolder
    )

def getOutputSize(useLandscape):
    outputWidth = 1920 if useLandscape else 1080
    outputHeight = 1080 if useLandscape else 1920
    return outputWidth, outputHeight

def getImageTemplate(templatesPath, className, useLandscape):
    relativeIdImagePath = os.path.join(templatesPath, f"{className}_{'landscape' if useLandscape else 'portrait'}.png")
    imageTemplatePath = os.path.abspath(relativeIdImagePath)
    imageTemplate = cv2.imread(imageTemplatePath, cv2.IMREAD_UNCHANGED)
    if imageTemplate.shape[2] != 4:
        raise ValueError(f"Template {className} must have an alpha channel for mask generation.")
    return imageTemplate

def getRotation(useLandscape):
    if useLandscape:
        return random.uniform(-10, 10)
    return random.uniform(-25, 25)

def getScale(useLandscape, rotation):
    if useLandscape and math.fabs(rotation) > 5:
        return random.uniform(0.5, 0.8)
    return random.uniform(0.5, 1.0)

def getRandomString(length=20):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choices(characters, k=length))

def getAllFilesRecursively(relativeFolder):
    allFiles = []
    for root, _, files in os.walk(os.path.abspath(relativeFolder)):
        for file in files:
            allFiles.append(os.path.join(root, file))
    return allFiles

def generateRealisticBackground(outputWidth, outputHeight, isBackgroundOnly, backgroundOnlyImagePath):
    backgroundType = random.choice([
        "wood", "plain", "textured", "table", "grid", "gradient",
        "noise", "paper", "concrete", "tiles"
    ])

    output_size = (outputHeight, outputWidth)

    if isBackgroundOnly and backgroundOnlyImagePath is not None:
        background = cv2.imread(backgroundOnlyImagePath)
        if background is not None:
            return background

        return np.full(output_size + (3,), [200, 200, 200], dtype=np.uint8)

    if backgroundType == "wood":
        baseColor = [random.randint(80, 140), random.randint(60, 110), random.randint(40, 80)]
        background = np.full(output_size + (3,), baseColor, dtype=np.uint8)
        for i in range(0, outputHeight, random.randint(3, 6)):
            intensity = random.randint(-10, 10)
            cv2.line(background, (0, i), (outputWidth, i),
                     [baseColor[0] + intensity, baseColor[1] + intensity, baseColor[2] + intensity], 1)
        return background

    elif backgroundType == "plain":
        baseColor = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]  # Full RGB range
        background = np.full(output_size + (3,), baseColor, dtype=np.uint8)
        noise = np.random.randint(-10, 10, output_size + (3,), dtype=np.int8)
        background = np.clip(background + noise, 0, 255).astype(np.uint8)
        return background

    elif backgroundType == "textured":
        baseColor = [random.randint(180, 220), random.randint(180, 220), random.randint(180, 220)]
        background = np.full(output_size + (3,), baseColor, dtype=np.uint8)
        for i in range(0, outputHeight, random.randint(8, 15)):
            cv2.line(background, (0, i), (outputWidth, i),
                     (baseColor[0] - 20, baseColor[1] - 20, baseColor[2] - 20), 1)
        for j in range(0, outputWidth, random.randint(8, 15)):
            cv2.line(background, (j, 0), (j, outputHeight),
                     (baseColor[0] - 20, baseColor[1] - 20, baseColor[2] - 20), 1)
        background = cv2.GaussianBlur(background, (5, 5), 0)
        return background

    elif backgroundType == "table":
        baseColor = [random.randint(200, 240), random.randint(200, 240), random.randint(200, 240)]
        background = np.full(output_size + (3,), baseColor, dtype=np.uint8)
        for i in range(0, outputHeight, random.randint(40, 80)):
            cv2.line(background, (0, i), (outputWidth, i),
                     (baseColor[0] - 40, baseColor[1] - 40, baseColor[2] - 40), 2)
        for j in range(0, outputWidth, random.randint(40, 80)):
            cv2.line(background, (j, 0), (j, outputHeight),
                     (baseColor[0] - 40, baseColor[1] - 40, baseColor[2] - 40), 2)
        return background

    elif backgroundType == "grid":
        baseColor = [random.randint(240, 255), random.randint(240, 255), random.randint(240, 255)]
        background = np.full(output_size + (3,), baseColor, dtype=np.uint8)
        for i in range(0, outputHeight, random.randint(50, 100)):
            cv2.line(background, (0, i), (outputWidth, i),
                     (baseColor[0] - 20, baseColor[1] - 20, baseColor[2] - 20), 1)
        for j in range(0, outputWidth, random.randint(50, 100)):
            cv2.line(background, (j, 0), (j, outputHeight),
                     (baseColor[0] - 20, baseColor[1] - 20, baseColor[2] - 20), 1)
        return background

    elif backgroundType == "gradient":
        startColor = [random.randint(100, 200), random.randint(100, 200), random.randint(100, 200)]
        endColor = [random.randint(200, 255), random.randint(200, 255), random.randint(200, 255)]
        background = np.zeros(output_size + (3,), dtype=np.uint8)
        for y in range(outputHeight):
            alpha = y / outputHeight
            color = [int(startColor[i] * (1 - alpha) + endColor[i] * alpha) for i in range(3)]
            background[y, :] = color
        return background

    elif backgroundType == "noise":
        background = np.random.randint(0, 256, output_size + (3,), dtype=np.uint8)
        if random.random() > 0.5:
            background = cv2.GaussianBlur(background, (random.choice([3, 5, 7]), random.choice([3, 5, 7])), 0)
        return background

    elif backgroundType == "paper":
        baseColor = [random.randint(220, 240), random.randint(220, 240), random.randint(220, 240)]
        background = np.full(output_size + (3,), baseColor, dtype=np.uint8)
        texture = np.random.randint(-10, 10, output_size + (3,), dtype=np.int8)
        background = np.clip(background + texture, 0, 255).astype(np.uint8)
        return background

    elif backgroundType == "concrete":
        baseColor = [random.randint(170, 190), random.randint(170, 190), random.randint(170, 190)]
        noise = np.random.randint(-30, 30, output_size + (3,), dtype=np.int8)
        background = np.clip(baseColor + noise, 0, 255).astype(np.uint8)
        background = cv2.GaussianBlur(background, (5, 5), 0)
        return background

    elif backgroundType == "tiles":
        tileSize = random.randint(50, 100)
        baseColor = [random.randint(150, 200), random.randint(150, 200), random.randint(150, 200)]
        background = np.full(output_size + (3,), baseColor, dtype=np.uint8)
        for i in range(0, outputHeight, tileSize):
            for j in range(0, outputWidth, tileSize):
                if random.random() > 0.5:
                    intensity = random.randint(-30, 30)
                    cv2.rectangle(background, (j, i), (j + tileSize, i + tileSize),
                                  (baseColor[0] + intensity, baseColor[1] + intensity, baseColor[2] + intensity), -1)
        return background

    return np.zeros(output_size + (3,), dtype=np.uint8)

def skrewImage(imageTemplate, imageTemplateWidth, imageTemplateHeight):
    srcPoints = np.float32([
        [0, 0],
        [imageTemplateWidth, 0],
        [imageTemplateWidth, imageTemplateHeight],
        [0, imageTemplateHeight]
    ])

    dstPoints = np.float32([
        [random.uniform(-50, 50), random.uniform(-50, 50)],
        [imageTemplateWidth + random.uniform(-50, 50), random.uniform(-50, 50)],
        [imageTemplateWidth + random.uniform(-50, 50), imageTemplateHeight + random.uniform(-50, 50)],
        [random.uniform(-50, 50), imageTemplateHeight + random.uniform(-50, 50)]
    ])

    perspectiveMatrix = cv2.getPerspectiveTransform(srcPoints, dstPoints)
    tiltedImage = cv2.warpPerspective(
        imageTemplate, perspectiveMatrix, (imageTemplateWidth, imageTemplateHeight),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )
    
    return tiltedImage, perspectiveMatrix

def rotateImage(tiltedImage, imageTemplateWidth, imageTemplateHeight, angle):
    center = (imageTemplateWidth // 2, imageTemplateHeight // 2)

    rotationMatrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(rotationMatrix[0, 0])
    sin = abs(rotationMatrix[0, 1])
    newWidth = int((imageTemplateHeight * sin) + (imageTemplateWidth * cos))
    newHeight = int((imageTemplateHeight * cos) + (imageTemplateWidth * sin))

    rotationMatrix[0, 2] += (newWidth / 2) - center[0]
    rotationMatrix[1, 2] += (newHeight / 2) - center[1]

    rotatedImageTemplate = cv2.warpAffine(
        tiltedImage, rotationMatrix, (newWidth, newHeight),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )
    
    return rotatedImageTemplate, rotationMatrix, angle

def addImageAugmentations(background):
    augmentations = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=40, p=0.5)
    ])
    augmented = augmentations(image=background)
    background = augmented["image"]
    return background

def moveValidationFiles(labelsTrainFolder, imagesValFolder, labelsValFolder, masksValFolder, percentage=0.1):
    folder = Path(labelsTrainFolder)
    allFiles = [f for f in folder.glob("*.*") if f.is_file()]
    sampleSize = max(1, int(len(allFiles) * percentage))
    sampleFiles = random.sample(allFiles, sampleSize)
    sampleFileNames = sorted([f.name.replace(".txt", "") for f in sampleFiles])
    originalImagesTrainFolder = labelsTrainFolder.replace("/labels/", "/images/")
    originalLabelsTrainFolder = labelsTrainFolder
    originalMasksTrainFolder = labelsTrainFolder.replace("/labels/", "/masks/")
    for sampleFileName in sampleFileNames:
        imageFile = os.path.join(originalImagesTrainFolder, f"{sampleFileName}.jpg")
        shutil.move(imageFile, imagesValFolder)
        print(f"{sampleFileName}.jpg moved to '{imagesValFolder}'")
        labelFile = os.path.join(originalLabelsTrainFolder, f"{sampleFileName}.txt")
        shutil.move(labelFile, labelsValFolder)
        print(f"{sampleFileName}.txt moved to '{labelsValFolder}'")
        if (masksValFolder is not None):
            maskFile = os.path.join(originalMasksTrainFolder, f"{sampleFileName}.jpg")
            shutil.move(maskFile, masksValFolder)
            print(f"{sampleFileName}.jpg moved to '{masksValFolder}'")

    print(f"Validation files moved to '/val' folders")

def writeDatasetConfig(datasetFolder, imagesTrainFolder, imagesValFolder, classNames):
    yamlFile = os.path.join(datasetFolder, "config.yaml")
    data = {
        "path": datasetFolder,
        "train": imagesTrainFolder,
        "val": imagesValFolder,
        "nc": len(classNames),
        "names": {i:label for i, label in enumerate(classNames)}
    }
    with open(yamlFile, "w") as fp:
        yaml.dump(data, fp, sort_keys=False)
    print(f"Dataset preparation complete. YAML file created at: {yamlFile}")
