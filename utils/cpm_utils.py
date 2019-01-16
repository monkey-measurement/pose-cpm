import numpy as np
import math
import cv2


M_PI = 3.1415967
H, W = 720, 1280
DEBUG = False


def gaussian_img(img_height, img_width, c_x, c_y, variance):
    gaussian_map = np.zeros((img_height, img_width))
    for x_p in range(img_width):
        for y_p in range(img_height):
            dist_sq = (x_p - c_x) * (x_p - c_x) + \
                      (y_p - c_y) * (y_p - c_y)
            exponent = dist_sq / 2.0 / variance / variance
            gaussian_map[y_p, x_p] = np.exp(-exponent)
    return gaussian_map


def read_image(file, boxsize):
    oriImg = cv2.imread(file)
    #oriImg = cv2.cvtColor(oriImg, cv2.COLOR_GRAY2RGB)

    scale = boxsize / (oriImg.shape[0] * 1.0)
    imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    output_img = np.ones((boxsize, boxsize, 3)) * 128

    img_h = imageToTest.shape[0]
    img_w = imageToTest.shape[1]
    if img_w < boxsize:
        offset = img_w % 2
        output_img[:, int(boxsize / 2 - math.floor(img_w / 2)):int(
            boxsize / 2 + math.floor(img_w / 2) + offset), :] = imageToTest
    else:
        output_img = imageToTest[:,
                     int(img_w / 2 - boxsize / 2):int(img_w / 2 + boxsize / 2), :]
    return output_img


def make_gaussian(size, fwhm=3, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 2.0 / fwhm / fwhm)


def make_gaussian_batch(heatmaps, size, fwhm):
    stride = heatmaps.shape[1] // size

    batch_datum = np.zeros(shape=(heatmaps.shape[0], size, size, heatmaps.shape[3]))

    for data_num in range(heatmaps.shape[0]):
        for joint_num in range(heatmaps.shape[3] - 1):
            heatmap = heatmaps[data_num, :, :, joint_num]
            center = np.unravel_index(np.argmax(heatmap), (heatmap.shape[0], heatmap.shape[1]))

            x = np.arange(0, size, 1, float)
            y = x[:, np.newaxis]

            if center is None:
                x0 = y0 = size * stride // 2
            else:
                x0 = center[1]
                y0 = center[0]

            batch_datum[data_num, :, :, joint_num] = np.exp(
                -((x * stride - x0) ** 2 + (y * stride - y0) ** 2) / 2.0 / fwhm / fwhm)
        batch_datum[data_num, :, :, heatmaps.shape[3] - 1] = np.ones((size, size)) - np.amax(
            batch_datum[data_num, :, :, 0:heatmaps.shape[3] - 1], axis=2)

    return batch_datum


def make_heatmaps_from_joints(input_size, heatmap_size, gaussian_variance, batch_joints):
    # Generate ground-truth heatmaps from ground-truth 2d joints
    scale_factor = input_size // heatmap_size
    batch_gt_heatmap_np = []
    for i in range(batch_joints.shape[0]):
        gt_heatmap_np = []
        invert_heatmap_np = np.ones(shape=(heatmap_size, heatmap_size))
        for j in range(batch_joints.shape[1]):
            cur_joint_heatmap = make_gaussian(heatmap_size,
                                              gaussian_variance,
                                              center=(batch_joints[i][j] // scale_factor))
            gt_heatmap_np.append(cur_joint_heatmap)
            invert_heatmap_np -= cur_joint_heatmap
        gt_heatmap_np.append(invert_heatmap_np)
        batch_gt_heatmap_np.append(gt_heatmap_np)
    batch_gt_heatmap_np = np.asarray(batch_gt_heatmap_np)
    batch_gt_heatmap_np = np.transpose(batch_gt_heatmap_np, (0, 2, 3, 1))

    return batch_gt_heatmap_np


def get_ground_truth_params(dataset_dir, line, img_size, heatmap_size, num_of_joints, gaussian_radius):
    line = line.split()

    # Read Image
    # TODO first line uncomment
    # cur_img_path = dataset_dir + '/' + line[0]
    cur_img_path = '../' + line[0]
    if DEBUG:
        print(cur_img_path)
    cur_img = cv2.imread(cur_img_path)
    try:
        assert cur_img.shape[2] == 3
    except:
        return [None]*5
        # corrupted_imgs += 1
        # corrupted_paths.append(cur_img_path)
        # continue

    cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)

    # Bounding Box
    tmp = [float(x) for x in line[1:5]]
    # (X_l, Y_l, X_h, Y_h)
    cur_hand_bbox = [min([tmp[1], tmp[3]]), min([tmp[0], tmp[2]]),max([tmp[1], tmp[3]]),max([tmp[0], tmp[2]])]
    if cur_hand_bbox[0] < 0: cur_hand_bbox[0] = 0
    if cur_hand_bbox[1] < 0: cur_hand_bbox[1] = 0
    if cur_hand_bbox[2] > cur_img.shape[1]: cur_hand_bbox[2] = cur_img.shape[1]
    if cur_hand_bbox[3] > cur_img.shape[0]: cur_hand_bbox[3] = cur_img.shape[0]

    # Keypoints
    cur_hand_joints_y = [float(i) for i in line[5:31:2]]
    cur_hand_joints_x = [float(i) for i in line[6:31:2]]
    vis = np.array([ 0 < cur_hand_joints_x[i] <= W and 0 < cur_hand_joints_y[i] <= H for i in range(num_of_joints) ])
    assert vis.sum() >= 10

    cur_img = cur_img[int(float(cur_hand_bbox[1])):int(float(cur_hand_bbox[3])),int(float(cur_hand_bbox[0])):int(float(cur_hand_bbox[2])),:]
    cur_hand_joints_x = [x - cur_hand_bbox[0] for x in cur_hand_joints_x]
    cur_hand_joints_y = [x - cur_hand_bbox[1] for x in cur_hand_joints_y]
    if DEBUG:
        print(cur_img.shape)
        print(cur_hand_joints_x)

    output_image = np.ones(shape=(img_size, img_size, 3)) * 128
    heatmap_output_image = np.ones(shape=(heatmap_size, heatmap_size, 3)) * 128
    output_heatmaps = np.zeros((heatmap_size, heatmap_size, num_of_joints))

    # Resize and pad image to fit output image size
    # if h > w
    if cur_img.shape[0] > cur_img.shape[1]:
        img_scale = img_size / (cur_img.shape[0] * 1.0)
        heatmap_scale = heatmap_size / (cur_img.shape[0] * 1.0)

        # Relocalize points
        cur_hand_joints_x = [x * heatmap_scale for x in cur_hand_joints_x]
        cur_hand_joints_y = [x * heatmap_scale for x in cur_hand_joints_y]

        # Resize image
        image = cv2.resize(cur_img, (0, 0), fx=img_scale, fy=img_scale, interpolation=cv2.INTER_LANCZOS4)
        offset = image.shape[1] % 2

        heatmap_image = cv2.resize(cur_img, (0, 0), fx=heatmap_scale, fy=heatmap_scale, interpolation=cv2.INTER_LANCZOS4)
        heatmap_offset = heatmap_image.shape[1] % 2

        output_image[:, int(img_size / 2 - math.floor(image.shape[1] / 2)): int(img_size / 2 + math.floor(image.shape[1] / 2) + offset), :] = image
        heatmap_output_image[:, int(heatmap_size / 2 - math.floor(heatmap_image.shape[1] / 2)): int(heatmap_size / 2 + math.floor(heatmap_image.shape[1] / 2) + heatmap_offset), :] = heatmap_image
        # cur_hand_joints_x = [x + (heatmap_size/2 - math.floor(heatmap_image.shape[1]/2)) for x in cur_hand_joints_x]

        cur_hand_joints_x = np.asarray(cur_hand_joints_x) + (heatmap_size/2 - math.floor(heatmap_image.shape[1]/2))
        cur_hand_joints_y = np.asarray(cur_hand_joints_y)

        if DEBUG:
            hmap = np.zeros((heatmap_size, heatmap_size))
            # Plot joints
            imshow(heatmap_output_image.astype(np.uint8)[:,:,[2,1,0]])
            plt.show()
            for i in range(num_of_joints):
                cv2.circle(heatmap_output_image, (int(cur_hand_joints_x[i]), int(cur_hand_joints_y[i])), 1, (0, 255, 0), 2)
                imshow(heatmap_output_image.astype(np.uint8)[:,:,[2,1,0]])
                plt.show()

                # Generate joint gaussian map
                part_heatmap = make_gaussian(heatmap_output_image.shape[0], gaussian_radius,[cur_hand_joints_x[i], cur_hand_joints_y[i]])
                hmap += part_heatmap * 50
        else:
            for i in range(num_of_joints):
                output_heatmaps[:, :, i] = make_gaussian(heatmap_size, gaussian_radius,[cur_hand_joints_x[i], cur_hand_joints_y[i]])

    else:
        img_scale = img_size / (cur_img.shape[1] * 1.0)
        heatmap_scale = heatmap_size / (cur_img.shape[1] * 1.0)

        # Relocalize points
        cur_hand_joints_x = [x * heatmap_scale for x in cur_hand_joints_x]
        cur_hand_joints_y = [x * heatmap_scale for x in cur_hand_joints_y]

        # Resize image
        image = cv2.resize(cur_img, (0, 0), fx=img_scale, fy=img_scale, interpolation=cv2.INTER_LANCZOS4)
        offset = image.shape[0] % 2

        heatmap_image = cv2.resize(cur_img, (0, 0), fx=heatmap_scale, fy=heatmap_scale, interpolation=cv2.INTER_LANCZOS4)
        heatmap_offset = heatmap_image.shape[0] % 2

        output_image[int(img_size / 2 - math.floor(image.shape[0] / 2)): int(img_size / 2 + math.floor(image.shape[0] / 2) + offset), :, :] = image
        heatmap_output_image[int(heatmap_size / 2 - math.floor(heatmap_image.shape[0] / 2)): int(heatmap_size / 2 + math.floor(heatmap_image.shape[0] / 2) + heatmap_offset), :, :] = heatmap_image
        # cur_hand_joints_y = map(lambda x: x + (heatmap_size / 2 - math.floor(heatmap_image.shape[0] / 2)),cur_hand_joints_y)

        cur_hand_joints_x = np.asarray(cur_hand_joints_x)
        cur_hand_joints_y = np.asarray(cur_hand_joints_y) + (heatmap_size/2 - math.floor(heatmap_image.shape[0]/2))
        
        if DEBUG:
            hmap = np.zeros((heatmap_size, heatmap_size))
            # Plot joints
            imshow(heatmap_output_image.astype(np.uint8)[:,:,[2,1,0]])
            plt.show()
            for i in range(num_of_joints):
                cv2.circle(heatmap_output_image, (int(cur_hand_joints_x[i]), int(cur_hand_joints_y[i])), 1, (0, 255, 0), 2)
                imshow(heatmap_output_image.astype(np.uint8)[:,:,[2,1,0]])
                plt.show()

                # Generate joint gaussian map
                part_heatmap = make_gaussian(heatmap_output_image.shape[0], gaussian_radius,[cur_hand_joints_x[i], cur_hand_joints_y[i]])
                hmap += part_heatmap * 50
        else:
            for i in range(num_of_joints):
                output_heatmaps[:, :, i] = make_gaussian(heatmap_size, gaussian_radius,[cur_hand_joints_x[i], cur_hand_joints_y[i]])
    if DEBUG:
        imshow(heatmap_output_image.astype(np.uint8)[:,:,[2,1,0]])
        plt.show()
        imshow(output_image.astype(np.uint8)[:,:,[2,1,0]])
        plt.show()
        # cv2.imshow('', hmap.astype(np.uint8)[:,:,[2,1,0]])
        # cv2.imshow('i', output_image.astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.imwrite("training_data/"+str(img_count)+".png", output_image.astype(np.uint8))

    # Create background map
    output_background_map = np.ones((heatmap_size, heatmap_size)) - np.amax(output_heatmaps, axis=2)
    output_heatmaps = np.concatenate((output_heatmaps, output_background_map.reshape((heatmap_size, heatmap_size, 1))),axis=2)
    #print(output_heatmaps.shape)
    '''
    cv2.imshow('', (output_background_map*255).astype(np.uint8))
    cv2.imshow('h', (np.amax(output_heatmaps[:, :, 0:21], axis=2)*255).astype(np.uint8))
    cv2.waitKey(1000)
    '''
    if DEBUG:
        pass
        # imshow((output_background_map*255).astype(np.uint8))
        # plt.show()
        # imshow((np.amax(output_heatmaps[:, :, 0:21], axis=2)*255).astype(np.uint8))
        # plt.show()

    return output_image, output_heatmaps, cur_hand_joints_x, cur_hand_joints_y, vis
