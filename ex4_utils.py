import sys

import numpy as np
import matplotlib.pyplot as plt


def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimun and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)
    return: Disparity map, disp_map.shape = Left.shape
    """
    min_r = disp_range[0]
    max_r = disp_range[1]
    disparity_map = np.zeros_like(img_l)
    for row in range(k_size, img_l.shape[0]-k_size):
        for col in range(k_size, img_l.shape[1]-k_size):
            window = img_l[row-k_size: row+k_size+1, col-k_size: col+k_size+1]
            max_SSD = 9999
            for i in range(max(col - max_r, k_size), min(col + max_r, img_l.shape[1] - k_size)):
                if abs(col-i) < min_r:
                    continue
                compere_win = img_r[row-k_size: row+k_size+1, i-k_size: i+k_size+1]
                ssd = ((window - compere_win)**2).sum()
                if ssd < max_SSD:
                    max_SSD = ssd
                    disparity_map[row, col] = abs(i - col)
    disparity_map *= (255//max_r)
    return disparity_map


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimun and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)
    return: Disparity map, disp_map.shape = Left.shape
    """
    disparity_map = np.zeros_like(img_l)
    for row in range(k_size, img_l.shape[0] - k_size):
        for col in range(k_size, img_l.shape[1] - k_size):
            window_l = img_l[row - k_size: row + k_size + 1, col - k_size: col + k_size + 1]
            norm_l = np.linalg.norm(window_l)
            sum_l = np.sqrt(np.sum((window_l-norm_l)**2))
            max_NC = -1000000
            disp = 0
            for i in range(disp_range[0], disp_range[1]):
                if (col + i - k_size >= 0) and (col + i + k_size + 1 < img_l.shape[1]) and\
                        (col - i - k_size >= 0) and (col - i + k_size + 1 < img_l.shape[1]):
                    # move right:
                    window_r = img_r[row - k_size: row + k_size + 1, col + i - k_size: col + i + k_size + 1]
                    norm_r = np.linalg.norm(window_r)
                    sum_r = np.sqrt(np.sum((window_r - norm_r) ** 2))
                    nnc = ((window_l-norm_l)*(window_r-norm_r)).sum() / (sum_l*sum_r)
                    if max_NC < nnc:
                        max_NC = nnc
                        disp = i
                    # move left:
                    window_r = img_r[row - k_size: row + k_size + 1, col - i - k_size: col - i + k_size + 1]
                    norm_r = np.linalg.norm(window_r)
                    sum_r = np.sqrt(np.sum((window_r - norm_r) ** 2))
                    nnc = np.sum((window_l - norm_l) * (window_r - norm_r)) / (sum_l * sum_r)
                    if max_NC < nnc:
                        max_NC = nnc
                        disp = i
            disparity_map[row, col] = disp
    disparity_map *= (255 // disp_range[1])
    return disparity_map


def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    """
    Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
    returns the homography and the error between the transformed points to their
    destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))
    src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
    dst_pnt: 4+ keypoints locations (x,y) on the destenation image. Shape:[4+,2]
    return: (Homography matrix shape:[3,3], Homography error)
    """
    index = 0
    A = np.zeros((8, 9))
    len_src = len(src_pnt)
    for i in range(0, len_src):
        x_dst = dst_pnt[i][0]
        y_dst = dst_pnt[i][1]
        x_src = src_pnt[i][0]
        y_src = src_pnt[i][1]
        A[index] = [x_src, y_src, 1, 0, 0, 0, -x_dst * x_src, -x_dst * y_src, -x_dst]
        A[index + 1] = [0, 0, 0, x_src, y_src, 1, -y_dst * x_src, -y_dst * y_src, -y_dst]
        index += 2
    A = np.asarray(A)
    U, D, V = np.linalg.svd(A)
    h = V[-1, :] / V[-1, -1]
    H = h.reshape(3, 3)

    # calculate the error:
    # first - padding src_pnt for legal matrices multiply
    new_col = [[1], [1], [1], [1]]
    padded_src = np.append(src_pnt, new_col, axis=1)
    pred = H.dot(padded_src.T).T

    pred_new = np.zeros((4, 2))
    for i in range(4):
        for j in range(2):
            pred_new[i][j] = pred[i][j] / pred[i][2]
    error = np.sqrt(np.square(pred_new - dst_pnt).mean())

    return H, error


def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
       Displays both images, and lets the user mark 4 or more points on each image. Then calculates the homography and transforms the source image on to the destination image. Then transforms the source image onto the destination image and displays the result.
       src_img: The image that will be 'pasted' onto the destination image.
       dst_img: The image that the source image will be 'pasted' on.
       output:
        None.
    """
    # for debug:
    # dst_p = np.array([[286, 333],
    #                  [1637, 183],
    #                  [1650, 688],
    #                  [290, 743]])
    # un sort points:
    # dst_p = np.array([[1637, 187],
    #                   [1629, 671],
    #                   [303, 760],
    #                   [290, 339]])

    dst_p = []
    fig1 = plt.figure()

    def onclick_1(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        dst_p.append([x, y])

        if len(dst_p) == 4:
            plt.close()
        plt.show()

    # display image 1
    cid = fig1.canvas.mpl_connect('button_press_event', onclick_1)
    plt.imshow(dst_img)
    plt.show()
    dst_p = np.array(dst_p)
    # dst_p = sort_points(dst_p)

    # 4 point in src image - the corners:
    point1 = [0, 0]
    point2 = [0, src_img.shape[1]-1]
    point3 = [src_img.shape[0]-1, src_img.shape[1]-1]
    point4 = [src_img.shape[0]-1, 0]

    src_p = np.array([point1, point2, point3, point4])

    h, e = computeHomography(dst_p, src_p)
    for x in range(dst_img.shape[0] - 1):
        for y in range(dst_img.shape[1] - 1):
            new_xy = h.dot(np.array([[x], [y], [1]]))
            norm = h[2, :].dot(np.array([[x], [y], [1]]))
            new_xy /= norm

            if 0 < new_xy[0] < src_img.shape[0] and 0 < new_xy[1] < src_img.shape[1]:
                dst_img[y][x][:] = src_img[int(new_xy[0]), int(new_xy[1]), :]

    plt.matshow(dst_img)
    plt.colorbar()
    plt.show()


# for un sorted points: (no need after task update)
def smaller_y(point1, point2):
    if point1[1] < point2[1]:
        return 1
    return 0


def smaller_x(point1, point2):
    if point1[0] < point2[0]:
        return 1
    return 0


def sort_points(points):
    # find the 2 upper points:
    if smaller_y(points[0], points[1]):
        miny_1 = points[0]
        miny_2 = points[1]
    else:
        miny_1 = points[1]
        miny_2 = points[0]

    for i in range(2, 4):
        if smaller_y(miny_2, points[i]):
            continue
        miny_2 = points[i]
        if smaller_y(miny_2, miny_1):
            # swap:
            miny_1 = miny_1 + miny_2
            miny_2 = miny_1 - miny_2
            miny_1 = miny_1 - miny_2

    # sort by x the upper points:
    new_p = []
    if smaller_x(miny_1, miny_2):
        dst1 = miny_1
        dst2 = miny_2
        new_p.append(dst1)
        new_p.append(dst2)
    else:
        dst1 = miny_2
        dst2 = miny_1
        new_p.append(dst1)
        new_p.append(dst2)

    # sort by x the lower points:
    lower = []
    for point in points:
        if(point[0] != dst1[0] or point[1] != dst1[1]) and (point[0] != dst2[0] or point[0] != dst2[0]):
            lower.append(point)

    if smaller_x(lower[0], lower[1]):
        dst3 = lower[0]
        dst4 = lower[1]
        new_p.append(dst3)
        new_p.append(dst4)
    else:
        dst3 = lower[1]
        dst4 = lower[0]
        new_p.append(dst3)
        new_p.append(dst4)

    return np.array(new_p)
