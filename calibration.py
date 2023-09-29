"""
----------------------------------------------------------------------------
Filename: calibration.py
Path: calibration.py
Date: 09.10.2023
Author: youngjae.you
Purpose: 카메라 캘리브레이션 테스트
History:
   - 
INFO:
    # My calibration chessboard is 9x7 squares, 40mm squares, 8x6 verticies
----------------------------------------------------------------------------
"""

import numpy as np
import cv2 as cv
import glob
import pickle
import os

# 체스보드 설정
chessboard_size = (8, 6) # 정점 갯수
chessboard_square_size = 40 # 40mm
frame_size = (640, 480) # 640x480 해상도

# 캘리브레이션 종료 조건
# 30은 30번 반복하겠다는 의미, 0.001은 정확도를 의미. 0.001보다 작으면 일찍 종료
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 3D 좌표 설정
# 체스보드의 3D 좌표를 설정한다. 체스보드의 3D 좌표는 (0,0,0), (1,0,0), (2,0,0) ... (7,5,0)으로 설정한다.
chessboard_3d_coordinates = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)
chessboard_3d_coordinates[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)

# 체스보드 한 칸의 크기
chessboard_3d_coordinates = chessboard_3d_coordinates * chessboard_square_size

# 모든 이미지에서의 3D 좌표를 저장할 배열
world_points = []
# 모든 이미지에서의 2D 좌표를 저장할 배열
image_points = []

# 이미지 로드
images = sorted(glob.glob('images/*.png'))

for image in images:
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
  
    # 체스보드 코너 찾기
    ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)
    
    if ret:
        world_points.append(chessboard_3d_coordinates)
        # 코너 위치를 세밀하게 조정할 때, 발견된 코너 주변의 11x11 픽셀 크기의 사각형 내에서 최적의 코너 위치를 검색한다.
        # 다시 말해서, 초기 코너 추정치에서 왼쪽으로 5 픽셀, 오른쪽으로 5 픽셀, 위로 5 픽셀, 아래로 5 픽셀을 보게 된다
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria) 
        image_points.append(corners2)
        
        # 코너 그리기
        cv.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(300)
        
        # find_corner 디렉토리를 만들고 거기에 코너가 그려진 이미지를 저장한다.
        cv.imwrite('find_corners/' + image.split(os.path.sep)[1], img)
        
    else:
        raise Exception("Chessboard not found in " + image)
    
cv.destroyAllWindows()

############## 캘리브레이션 ##############
ret, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = cv.calibrateCamera(world_points, image_points, frame_size, None, None)

# 3D -> 2D 좌표 변환
print("Camera matrix: \n" + str(camera_matrix))
print("Distortion coefficients: \n" + str(distortion_coefficients))

# 캘리브레이션 결과를 저장
pickle.dump((camera_matrix, distortion_coefficients), open( "calibration.pkl", "wb" ))
pickle.dump(camera_matrix, open( "cameraMatrix.pkl", "wb" ))
pickle.dump(distortion_coefficients, open( "dist.pkl", "wb" ))

############## 왜곡 제거 ##############
img = cv.imread('sample.png')
h, w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w,h), 1, (w,h))

print("New camera matrix: \n" + str(newCameraMatrix))
# ROI는 왜곡되지 않은 이미지 중 어느 부분이 유효하고 유지되어야 하는지를 지정하는 직사각형.
# (x, y, 너비, 높이) 형식으로 제공
print("ROI: \n" + str(roi))


cv.undistort(img, camera_matrix, distortion_coefficients, None, newCameraMatrix)

