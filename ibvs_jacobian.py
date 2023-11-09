import numpy as np

def ibvs_jacobian(K, pt, z):
    """
    Determine the Jacobian for IBVS.

    The function computes the Jacobian for image-based visual servoing,
    given the camera matrix K, an image plane point, and the estimated
    depth of the point. The x and y focal lengths in K are guaranteed 
    to be identical.

    Parameters:
    -----------
    K  - 3x3 np.array, camera intrinsic calibration matrix.
    pt - 2x1 np.array, image plane point. 
    z  - Scalar depth value (estimated).

    Returns:
    --------
    J  - 2x6 np.array, Jacobian. The matrix must contain float64 values.
    """

    #--- FILL ME IN ---
    f=K[0][0]
    ubar=pt[0][0]-K[0][2]
    vbar=pt[1][0]-K[1][2]
    J=np.array([[-f/z, 0, ubar/z, (ubar*vbar)/f, -((f**2)+(ubar**2))/f, vbar],
                [0, -f/z, vbar/z, ((f**2)+(vbar**2))/f, -(ubar*vbar)/f, -ubar]])
    


    #------------------

    correct = isinstance(J, np.ndarray) and \
        J.dtype == np.float64 and J.shape == (2, 6)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return J
