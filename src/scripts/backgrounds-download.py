import os
import shutil
import subprocess

def downloadResourcesFiles(bboxes_filename, bboxes_filepath, downloader_filename, downloader_filepath):
    if not os.path.isfile(bboxes_filepath):
        wget_url = f"https://storage.googleapis.com/openimages/v6/{bboxes_filename}"
        subprocess.run(["wget", wget_url, "-O", bboxes_filepath])
    if not os.path.isfile(downloader_filepath):
        wget_url = f"https://raw.githubusercontent.com/openimages/dataset/master/{downloader_filename}"
        subprocess.run(["wget", wget_url, "-O", downloader_filepath])

def getAllClassIds(classes_file, class_names):
    id_name_dict = {}
    with open(classes_file, 'r') as f:
        for line in f:
            id, label = line.split(',')
            label = label.strip()
            if label in class_names:
                id_name_dict[label] = id
        
    return id_name_dict

def getKeyByValue(d, targetValue):
    for key, value in d.items():
        if value == targetValue:
            return key
    return None

def downloadOpenImagesDataset(script_path, image_list_file_path, output_folder):
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    params = [image_list_file_path, f"--download_folder={output_folder}", "--num_processes=5"]
    subprocess.run(["python", script_path] + params)

if __name__ == "__main__":
    basePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    openimagesPath = os.path.join(basePath, "openimages")
    backgroundsPath = os.path.join(openimagesPath, "backgrounds")
    if os.path.isdir(backgroundsPath):
        shutil.rmtree(backgroundsPath)

    bboxesFilename = "oidv6-train-annotations-bbox.csv"
    bboxesFilepath = os.path.join(openimagesPath, bboxesFilename)
    downloaderFileName = "downloader.py"
    downloaderFilePath = os.path.join(openimagesPath, downloaderFileName)
    classDescriptionsFilePath = os.path.join(openimagesPath, 'class-descriptions-boxable.csv')

    downloadResourcesFiles(bboxesFilename, bboxesFilepath, downloaderFileName, downloaderFilePath)

    names = ['Bottle', 'Chair', 'Computer keyboard', 'Desk', 'Headphones', 'Laptop', 'Printer', 'Table']
    allClassIds = getAllClassIds(classDescriptionsFilePath, names)
    print(allClassIds)

    imageListFileList = []
    classesDic = {}

    with open(bboxesFilepath, 'r') as f:
        line = f.readline()
        while len(line) != 0:
            id, _, className, _, x1, x2, y1, y2, _, _, _, _, _ = line.split(',')[:13]
            realClassName = getKeyByValue(allClassIds, className)

            if (realClassName != None):
                if realClassName not in classesDic:
                    classesDic[realClassName] = 0
                    os.makedirs(os.path.join(backgroundsPath, realClassName), exist_ok=True)

                if classesDic[realClassName] < 1000 and className in list(allClassIds.values()) and id not in imageListFileList:
                    imageListFileList.append(id)
                    imageListFilePath = os.path.join(backgroundsPath, f'image_list_file-{realClassName}.txt')
                    with open(f"{os.path.join(imageListFilePath, )}", 'a') as fw:
                        fw.write('{}/{}\n'.format('train', id))
                        classesDic[realClassName] = classesDic[realClassName] + 1
                        print(f"Adding entry {id} for {realClassName}")
                else:
                    print(f"Skipped entry {id} for {realClassName}")
            line = f.readline()
        f.close()

    for realClassName in list(allClassIds.keys()):
        imageListFilePath = os.path.join(backgroundsPath, f'image_list_file-{realClassName}.txt')
        backgroundPath = os.path.join(backgroundsPath, realClassName)
        if os.path.isdir(backgroundPath):
            fileCount = len([name for name in os.listdir(backgroundPath) if os.path.isfile(os.path.join(backgroundPath, name))])
            if fileCount == 0:
                downloadOpenImagesDataset(downloaderFilePath, imageListFilePath, backgroundPath)
        os.unlink(imageListFilePath)
