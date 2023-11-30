import cv2
import numpy as np
import glob

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
    
    def __init__(self, size, markerOrientation):
                
        self.objectPoints = dict()
        for markerID, orientation in markerOrientation.items():
            
            # find the ArUco marker coordinates in the block coordinate frame
            z_face, x_face = orientation
            y_face = np.cross(z_face, x_face)
            
            objectPoints = (size/2) * ( [ -x_face + y_face, x_face + y_face, 
                                         x_face - y_face, -x_face - y_face] + z_face)
            
            self.objectPoints[markerID] = objectPoints      


def getRpfromImage(image, cameraMatrix, blocks, arucoDict=cv2.aruco.DICT_4X4_1000):
    """Find the homogenous transform matricies from the camera to each block
    
    image: an image containing blocks
    cameraMatrix: the intrinsic matrix of the camera
    blocks: a list of Block objects whose homogenous transform matricies is to be found
    arucoDict: the ArUco dict of the block's ArUco markers
    """
    
    arucoDict = cv2.aruco.Dictionary_get(arucoDict)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

    imagePoints = dict()

    if len(corners) > 0:

        ids = ids.flatten()

        for (markerCorner, markerID) in zip(corners, ids):

            print(f"[INFO] ArUco marker ID: {markerID}")
            imagePoints[markerID] = markerCorner.reshape((4, 2))

    else:
        print("Warning: No corners found.")


    R = dict()
    p = dict()
    for block in blocks:
        X = []
        y = []
        for markerID in block.objectPoints:

            if markerID not in imagePoints:
                continue

            X.append(block.objectPoints[markerID])
            y.append(imagePoints[markerID])

        if len(X) == 0:
            continue

        X = np.concatenate(X)  # TODO: write better names for X, y
        y = np.concatenate(y)
        
        ret, rvecs, tvecs = cv2.solvePnP(X, y, cameraMatrix, None)

        R[block], _ = cv2.Rodrigues(rvecs)
        p[block] = tvecs
        
    return R, p


if __name__ == "__main__":

    print("hoping to detect a block")


    block1 = Block(0.03, { 0: [ Directions.z,  Directions.y],
			               1: [-Directions.z, -Directions.x],
                           2: [ Directions.x,  Directions.y],
                           3: [-Directions.y,  Directions.x],
                           4: [-Directions.x, -Directions.y],
                           5: [ Directions.y, -Directions.x]})

    block2 = Block(0.03, { 6: [ Directions.z,  Directions.y],
			               7: [-Directions.z, -Directions.x],
                           8: [ Directions.x,  Directions.y],
                           9: [-Directions.y,  Directions.x],
                          10:[-Directions.x, -Directions.y],
                          11:[ Directions.y, -Directions.x]})

    block3 = Block(0.03, {12:[ Directions.z,  Directions.y],
			              13:[-Directions.z, -Directions.x],
                          14:[ Directions.x,  Directions.y],
                          15:[-Directions.y,  Directions.x],
                          16:[-Directions.x, -Directions.y],
                          17:[ Directions.y, -Directions.x]})

    block4 = Block(0.03, {18:[ Directions.z,  Directions.y],
                          19:[-Directions.z, -Directions.x],
                          20:[ Directions.x,  Directions.y],
                          21:[-Directions.y,  Directions.x],
                          22:[-Directions.x, -Directions.y],
                          23:[ Directions.y, -Directions.x]})

    blocks = [block1, block2, block3, block4]



    cameraMatrix = np.array([[ 9.551e3, 0, 4.085e2],
                             [ 0, 1.209e4, -6.502 ],
                             [ 0, 0, 1 ]])

    # find camera port
    for port in range(10):
        camera = cv2.VideoCapture(port)
        ret, image = camera.read()
        if ret:
            break


    R, p = getRpfromImage(image, cameraMatrix, blocks)
    print(R)
    print(p)

    
