import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

class Directions:
        x = np.array([1, 0, 0])
        y = np.array([0, 1, 0])
        z = np.array([0, 0, 1])
        
class Block:
    """A class used to represent a block with ArUco markers as faces
    
    Attributes
    ----------
    size : float
        The length of each edge on the block in meters (we assume a perfect cube)
        
    objectPoints : dict[int: [np.ndarray]]
        The four corners of each ArUco marker.  
        The keys are the ArUco marker ids and values are a list of 3D corrordinates of each marker 
        in the order [top left, top right, bottom right, bottom left]
    
    """
    
    def __init__(self, size:float, markerOrientation: dict[int: [np.ndarray, np.ndarray]]):
                
        self.objectPoints = dict()
        for markerID, orientation in markerOrientation.items():
            
            # find the ArUco marker coordinates in the block coordinate frame
            z_face, x_face = orientation
            y_face = np.cross(z_face, x_face)
            
            objectPoints = (size/2) * ( [ -x_face + y_face, x_face + y_face, 
                                         x_face - y_face, -x_face - y_face] + z_face)
            
            self.objectPoints[markerID] = objectPoints

def getHomogenousTransformsMatricies(image, cameraMatrix, blocks, arucoDict=cv2.aruco.DICT_4X4_1000):
    """Find the homogenous transform matricies from the camera to each block
    
    image: an image containing blocks
    cameraMatrix: the intrinsic matrix of the camera
    blocks: a list of Block objects whose homogenous transform matricies is to be found
    arucoDict: the ArUco dict of the block's ArUco markers
    """
    
    arucoDict = cv2.aruco.Dictionary_get(arucoDict)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

    corners_dict = dict()

    if len(corners) > 0:

        ids = ids.flatten()

        for (markerCorner, markerID) in zip(corners, ids):

            print(f"[INFO] ArUco marker ID: {markerID}")
            corners_dict[markerID] = markerCorner.reshape((4, 2))

    else:
        print("Warning: No corners found.")


    homogenousTransformsMatricies = dict()
    for block in [block1]:
        X = []
        y = []
        for markerID in block.objectPoints:

            if markerID not in corners_dict:
                continue

            X.append(block.objectPoints[markerID])
            y.append(corners_dict[markerID])

        X = np.concatenate(X)  # TODO: write better names for X, y (and corners_dict)
        y = np.concatenate(y)
        
        ret, rvecs, t = cv2.solvePnP(X, y, cameraMatrix, None)

        R, _ = cv2.Rodrigues(rvecs)
        H = np.concatenate([R, t], axis=1)
        
        homogenousTransformsMatricies[block] = H
        
    return homogenousTransformsMatricies
