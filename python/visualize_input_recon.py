import numpy as np
import cv2
import os

kInputFileName = "007-07-0695-060-10-1116_06-307_61-155_57730897025192-0"
kInputBasePath = "D:/Workspace/Dataset/ETRI/GarbageDumping/action_data/60_10"
kReconBasePath = "D:/Workspace/Dataset/ETRI/GarbageDumping/action_data/60_10/recon"
kResultBasePath = "D:/Workspace/Dataset/ETRI/GarbageDumping/action_data/60_10/video"

# kOriginCoord = 100
kOriginCoord = 0
kImageSize = 600

kLimbs = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6],
         [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
         [11, 12], [12, 13], [1, 0], [0, 14], [14, 16],
         [0, 15], [15, 17]]


def draw_keypoints(img, keypoints, confidences, color):
    xs = keypoints[0::2]
    ys = keypoints[1::2]
    for limb in kLimbs:
        if 0 == confidences[limb[0]] * confidences[limb[1]]:
            continue
        point_1 = (int(xs[limb[0]]), int(ys[limb[0]]))
        point_2 = (int(xs[limb[1]]), int(ys[limb[1]]))
        cv2.line(img, point_1, point_2, color, thickness=3)

    for ptIdx in range(len(xs)):
        if 0 == confidences[ptIdx]:
            continue
        center = (int(xs[ptIdx]), int(ys[ptIdx]))
        cv2.circle(img, center, 4, color=color, thickness=-1)

    return img


if __name__ == "__main__":

    input_sample = np.load(os.path.join(kInputBasePath, kInputFileName + '.npy'))
    recon_sample = np.load(os.path.join(kReconBasePath, kInputFileName + '_recon.npy'))

    # rescale and translation
    non_zero_input = np.array([val if val != 0 else kOriginCoord for val in input_sample.flatten()])
    non_zero_recon = np.array([val if val != 0 else kOriginCoord for val in recon_sample.flatten()])
    max_pos = max(np.amax(abs(non_zero_input - kOriginCoord)), np.amax(abs(non_zero_recon - kOriginCoord)))
    min_pos = min(np.amin(abs(non_zero_input - kOriginCoord)), np.amin(abs(non_zero_recon - kOriginCoord)))
    pos_range = max_pos - min_pos
    input_sample_adjusted = (input_sample - kOriginCoord) / pos_range * (0.5 * kImageSize - 1) + 0.5 * kImageSize
    recon_sample_adjusted = (recon_sample - kOriginCoord) / pos_range * (0.5 * kImageSize - 1) + 0.5 * kImageSize

    # video writer
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    video_out = cv2.VideoWriter(os.path.join(kResultBasePath, kInputFileName + '.avi'), fourcc, 30.0, (kImageSize, kImageSize))

    for i in range(input_sample.shape[0]):

        xs = input_sample[i, 0::2]
        ys = input_sample[i, 1::2]
        confidences = [1] * len(xs)
        for j in range(len(xs)):
            if 0 == xs[j] + ys[j]:
                confidences[j] = 0

        img = np.zeros((kImageSize, kImageSize, 3), np.uint8)
        img = draw_keypoints(img, input_sample_adjusted[i, :], confidences, (0, 0, 255))
        img = draw_keypoints(img, recon_sample_adjusted[i, :], confidences, (0, 255, 0))
        video_out.write(img)

        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_out.release()
    cv2.destroyAllWindows()
