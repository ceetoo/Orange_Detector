
#Orange Detector
#A project to detect oranges on fruit trees in images
#Written as part of the Stanford CS106A - Code in Place course
#Author: ceetoo
#https://github.com/ceetoo/Orange_Detector
#Date: 2020-05-24

import math
import numpy as np
from scipy import ndimage
from skimage import io, morphology, img_as_ubyte, img_as_float
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

N_COLOURS = 7
MIN_RAD = 23
MAX_RAD = 40
DIST_RAD = int(0.9 * MIN_RAD)
ORGKMN = [0.7, 0.4, 0.03]


def main():
    print('Find the oranges ...')
    for p in range(1, 7):
        orange_detector(p)


# function to detect oranges in images
def orange_detector(picnum):
    picname = ''.join(['Oranges/', str(picnum), '.jpg'])
    img = io.imread(picname)
    io.imsave('input'.join([str(picnum), '.jpg']), img_as_ubyte(img))
    h, w, d = tuple(img.shape)

    img_arr = np.reshape(img_as_float(img), (w * h, d))
    img_arr_sample = shuffle(img_arr, random_state=0)[:max(w, h)]
    kmeans = KMeans(n_clusters=N_COLOURS, random_state=0).fit(img_arr_sample)
    labels = kmeans.predict(img_arr)

    org_lbl = 0
    labelcol = kmeans.cluster_centers_[0, :]
    distmin = math.sqrt((labelcol[0] - ORGKMN[0]) ** 2 + (labelcol[1] - ORGKMN[1]) ** 2 + (labelcol[2] - ORGKMN[2]) ** 2)
    for i in range(1, N_COLOURS):
        labelcol = kmeans.cluster_centers_[i, :]
        dist = math.sqrt((labelcol[0] - ORGKMN[0]) ** 2 + (labelcol[1] - ORGKMN[1]) ** 2 + (labelcol[2] - ORGKMN[2]) ** 2)
        if dist < distmin:
            distmin = dist
            org_lbl = i

    img_kmn, mask_org = clusters_to_RGB(kmeans.cluster_centers_, labels, org_lbl, w, h, d)
    io.imsave('kmeans'.join([str(picnum), '.jpg']), img_as_ubyte(img_kmn))
    io.imsave('orangeareas'.join([str(picnum), '.jpg']), img_as_ubyte(mask_org))

    mask_org = morphology.remove_small_objects(mask_org, min_size=MIN_RAD*10, connectivity=2)
    mask_org = ndimage.binary_fill_holes(mask_org)
    io.imsave('mask'.join([str(picnum), '.jpg']), img_as_ubyte(mask_org))

    img2 = np.array(img)
    img2[~mask_org] = 0
    io.imsave('candidates'.join([str(picnum), '.jpg']), img_as_ubyte(img2))

    img_edges = canny(mask_org, sigma=3)
    hough_radii = np.arange(MIN_RAD, MAX_RAD)
    hough_res = hough_circle(img_edges, hough_radii)

    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, min_xdistance=DIST_RAD, min_ydistance=DIST_RAD)
    for i in range(radii.shape[0]):
        curr_circ = mask_circle(cx[i], cy[i], radii[i], w, h)
        fill_area = np.sum(curr_circ)
        org_area = np.sum(mask_org[curr_circ])
        if org_area >= 0.4*fill_area:
            circ_border = draw_circle(cx[i], cy[i], radii[i] - 5, 5, w, h)
            img[circ_border] = (20, 20, 255)
    io.imsave('oranges'.join([str(picnum), '.jpg']), img_as_ubyte(img))


# draw a circle around the detected oranges, border thickness is user specified
def draw_circle(cx, cy, r_in, border, w, h):
    r_out = (r_in + border)
    ri2 = r_in**2
    ro2 = r_out**2
    mask = np.zeros((h, w), dtype=bool)
    for x in range(max(0, cx - r_out), min(w - 1, cx + r_out)):
        x2 = (x - cx)**2
        for y in range(max(0, int(cy - math.sqrt(ro2 - x2))), min(h - 1, int(cy + math.sqrt(ro2 - x2)))):
            yo = y - cy
            if x2 <= ri2:
                ri_sq = math.sqrt(ri2 - x2)
                if yo >= ri_sq or yo <= -ri_sq:
                    mask[y, x] = 1
            else:
                mask[y, x] = 1
    return mask


# create a mask of the interior region of the circle defining a candidate orange
def mask_circle(cx, cy, radius, w, h):
    r2 = radius**2
    mask = np.zeros((h, w), dtype=bool)
    for x in range(max(0, cx - radius), min(w - 1, cx + radius)):
        x2 = (x - cx)**2
        for y in range(max(0, int(cy - math.sqrt(r2 - x2))), min(h - 1, int(cy + math.sqrt(r2 - x2)))):
            if x2 <= r2:
                mask[y, x] = 1
    return mask


# create RGB image from kmeans clusters with each label colored with the cluster mean color
def clusters_to_RGB(cluster_colors, lbls, lbl_seek, w, h, d):
    image = np.zeros((h, w, d))
    mask = np.zeros((h, w), dtype=bool)
    lbl_idx = 0
    for y in range(h):
        for x in range(w):
            curr_lbl = lbls[lbl_idx]
            image[y, x] = cluster_colors[curr_lbl]
            if curr_lbl == lbl_seek:
                mask[y, x] = 1
            lbl_idx += 1
    return image, mask


if __name__ == '__main__':
    main()