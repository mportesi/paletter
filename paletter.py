import sklearn.cluster
import os
import cv2
import numpy as np
import argparse

offset = 0  # to implement later..
filename = ""
FOLDERNAME = "\generated_palette"

def plot_colors(centroids):
    # bar charts
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # Sort to get gradient look
    centroids = sorted(centroids, key=lambda x: sum(x))

    for color in centroids[offset:]:
        # create n=clusters number of color rectangles
        new_length = 300 - margin * (clusters - 1)
        endX = startX + new_length/clusters
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)
        cv2.rectangle(bar, (int(endX), 0), (int(endX + margin), 50), (255, 255, 255), -1)
        startX = endX + margin

    return bar

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def create_palette(imagePath, Clusters=-1, Bordersize=-1, Margin=-1, paletteOnly = False):
    global clusters, borderSize, offset, margin
    if Clusters > -1:
        clusters = Clusters
    if Bordersize > -1:
        borderSize = Bordersize
    if Margin > -1:
        margin = Margin

    # get OpenCV image
    image = cv2.imread(imagePath)
    imgWidth = image.shape[1]

    # image copy for work-use
    imageCopy = image_resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width=100)

    # reshaping to smaller image to speed-up k-means algo
    pixelImage = imageCopy.reshape((imageCopy.shape[0] * imageCopy.shape[1], 3))

    # sklearn K-Means algorithm to create the clusters of colors
    clt = sklearn.cluster.KMeans(n_clusters=clusters+offset)
    clt.fit(pixelImage)

    # create the palette
    bar = plot_colors(clt.cluster_centers_)

    # color bar width equal to image width
    barImage = image_resize(cv2.cvtColor(bar, cv2.COLOR_RGB2BGR), width=int(imgWidth))

    # Separator between the image and the color bar
    whiteSpace = np.zeros((int(borderSize/2), int(imgWidth), 3), np.uint8)
    cv2.rectangle(whiteSpace, (0, 0), (int(imgWidth), int(borderSize/2)),
                  (255, 255, 255), -1)

    filename = os.path.basename(imagePath)[:-4]
    basepath = os.getcwd()+FOLDERNAME + "\\"+ filename
    temp_id = 0

    if not paletteOnly:
        # combine image and palette
        newImg = np.concatenate([image, whiteSpace, barImage], axis=0)
        # outline border
        newImgShape = newImg.shape
        w = newImgShape[1]
        h = newImgShape[0]
        base_size = h+borderSize, w+borderSize, 3
        base = np.zeros(base_size, dtype=np.uint8)
        cv2.rectangle(
            base,
            (0, 0),
            (w + borderSize, h + borderSize),
            (255, 255, 255), borderSize)
        base[
            (int(borderSize/2)):h+(int(borderSize/2)),
            (int(borderSize/2)):w+(int(borderSize/2))
            ] = newImg

        # all done.
        while os.path.exists(basepath + "_palette_%s.png" % temp_id):
            temp_id += 1
        cv2.imwrite(basepath+ "_palette_%s.png" % temp_id, base)

    else:
        # outline border
        newImgShape = barImage.shape
        w = newImgShape[1]
        h = newImgShape[0]
        base_size = h+borderSize, w+borderSize, 3
        base = np.zeros(base_size, dtype=np.uint8)
        cv2.rectangle(
            base,
            (0, 0),
            (w + borderSize, h + borderSize),
            (255, 255, 255), borderSize)
        base[
            (int(borderSize/2)):h+(int(borderSize/2)),
            (int(borderSize/2)):w+(int(borderSize/2))
            ] = barImage

        # all done.
        while os.path.exists(basepath + "_palette_%s.png" % temp_id):
            temp_id += 1
        cv2.imwrite(basepath+ "_palette_%s.png" % temp_id, base)


def main():
    if not os.path.isdir(os.getcwd()+FOLDERNAME):
        os.mkdir(os.getcwd()+FOLDERNAME)

    parser = argparse.ArgumentParser(description="Generate a palette from an image file.")
    parser.add_argument("filepath", metavar='FILEPATH', type=str, help="Path of image to use.")
    parser.add_argument("-n", type=int, default=9, help="Number of color to plot in the palette. Default 9")
    parser.add_argument("--border", type=int, default=20, help="Size of the border. Default 20.")
    parser.add_argument("--margin", type=int, default=2, help="Size of the margin between colors. Default 2.")
    parser.add_argument("-p", "--palette", type=int, default=0,help="Generate only the palette. Value 0,1. Default 0.")
    args = parser.parse_args()
    filepath = args.filepath

    if not os.path.isfile(filepath):
        print("The path entered is not valid.")
    else:
        clustersNumber = args.n
        borderSize = args.border
        margin = args.margin
        paletteOnly = args.palette
        create_palette(filepath, clustersNumber, borderSize, margin, paletteOnly)

if __name__ == "__main__":
    main()