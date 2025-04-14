import cv2
import numpy as np

def calibrate_from_images_using_remapping(images, rows=12, cols=12):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []
    corner_images = []
    gray_shape = None
    sample_img = None

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape[::-1]
        sample_img = img.copy()  # Keep a reference image for undistortion
        ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)

        if ret:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners)
            drawn = cv2.drawChessboardCorners(img.copy(), (rows, cols), corners, ret)
            corner_images.append(drawn)

    if objpoints:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)
        np.savez('calib_npz_remap.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            total_error += cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error = total_error / len(objpoints)

        # Undistortion using the first image
        h, w = sample_img.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # Method 1: Using cv2.undistort
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, new_camera_mtx, (w, h), 5)
        dst2 = cv2.remap(sample_img, mapx, mapy, cv2.INTER_LINEAR)

        # Crop the image using ROI
        x, y, w, h = roi
        cropped_remapped = dst2[y:y+h, x:x+w]

        return {
            "camera_matrix": mtx,
            "dist_coeffs": dist,
            "rvecs": rvecs,
            "tvecs": tvecs,
            "error": mean_error,
            "images": corner_images,
            "original": sample_img,
            "remapped": dst2,
            "remapped_cropped": cropped_remapped,
            "calibration_file": "calib_npz_remap is saved"
        }

    return None
