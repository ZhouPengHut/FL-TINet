import glob
import random
import xml.etree.ElementTree as ET

import numpy as np


def cas_iou(box,cluster):
    x = np.minimum(cluster[:,0],box[0])
    y = np.minimum(cluster[:,1],box[1])

    intersection = x * y
    area1 = box[0] * box[1]

    area2 = cluster[:,0] * cluster[:,1]
    iou = intersection / (area1 + area2 -intersection)

    return iou


def calculate_IoU(box, box_gt):
    # Calculate the intersection area
    intersection = np.maximum(0, np.minimum(box[2], box_gt[2]) - np.maximum(box[0], box_gt[0])) * np.maximum(0,
                                                                                                             np.minimum(
                                                                                                                 box[3],
                                                                                                                 box_gt[
                                                                                                                     3]) - np.maximum(
                                                                                                                 box[1],
                                                                                                                 box_gt[
                                                                                                                     1]))

    # Calculate the union area
    union = (box[2] - box[0]) * (box[3] - box[1]) + (box_gt[2] - box_gt[0]) * (box_gt[3] - box_gt[1]) - intersection

    # Calculate IoU
    IoU = intersection / union
    return IoU

def calculate_loss(box, box_gt, rho_square, C, alpha):
    w, h = box[2] - box[0], box[3] - box[1]
    w_gt, h_gt = box_gt[2] - box_gt[0], box_gt[3] - box_gt[1]

    # Calculate nu using Equation (6)
    nu = (4 / np.pi ** 2) * (np.arctan(w_gt / h_gt) - np.arctan(w / h)) ** 2

    # Calculate IoU using Equation (7)
    IoU = calculate_IoU(box, box_gt)

    # Calculate loss using Equation (5)
    loss = 1 - IoU + (rho_square / C ** 2) + alpha * nu

    return loss

def avg_iou(box,cluster):
    return np.mean([np.max(cas_iou(box[i],cluster)) for i in range(box.shape[0])])


def dk_means(X, n, sigma, delta):
    # Generate initial cluster centers randomly
    S = X[np.random.choice(X.shape[0], n, replace=False)]

    # Initialize TD
    TD = float('inf')

    while TD > delta:
        TD = 0
        for i, xi in enumerate(X):
            # Calculate the distances to all cluster centers
            distances = [distance_formula(xi, muc, sigma) for muc in S]

            # Find the index of the closest cluster center
            v = np.argmin(distances)

            # Update TD
            TD += distances[v]

            # Move xi to the closest cluster
            S[v] = (S[v].astype(int) * len(np.where(S == S[v])[0]) + xi.astype(int)).astype(float) / (len(np.where(S == S[v])[0]) + 1)

        # Update the cluster center S
        for f in range(n):
            S[f] = np.mean(X[np.where(S == S[f])[0]], axis=0)

    return S



def load_data(path):
    data = []
    # For each XML file, search for the bounding box(es)
    for xml_file in glob.glob('{}/*xml'.format(path)):
        tree = ET.parse(xml_file)
        height = int(tree.findtext('./size/height'))
        width = int(tree.findtext('./size/width'))
        if height<=0 or width<=0:
            continue
        
        # For each object, obtain its width and height
        for obj in tree.iter('object'):
            xmin = int(float(obj.findtext('bndbox/xmin'))) / width
            ymin = int(float(obj.findtext('bndbox/ymin'))) / height
            xmax = int(float(obj.findtext('bndbox/xmax'))) / width
            ymax = int(float(obj.findtext('bndbox/ymax'))) / height

            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            # Obtain the width and height
            data.append([xmax-xmin,ymax-ymin])
    return np.array(data)


if __name__ == '__main__':


    SIZE = 416
    anchors_num = 9
    # Load the dataset, which can utilize the XML format from VOC
    path = r'./VOCdevkit/VOC2007/Annotations'
    sigma = 0

    # Storage format converted to scaled width, height
    data = load_data(path)

    delta=1
    
    # use dk-means
    out = dk_means(data,anchors_num, sigma, delta)
    out = out[np.argsort(out[:,0])]
    print('acc:{:.2f}%'.format(avg_iou(data,out) * 100))
    print(out*SIZE)
    data = out*SIZE
    f = open("yolo_anchors.txt", 'w')
    row = np.shape(data)[0]
    for i in range(row):
        if i == 0:
            x_y = "%d,%d" % (data[i][0], data[i][1])
        else:
            x_y = ", %d,%d" % (data[i][0], data[i][1])
        f.write(x_y)
    f.close()
