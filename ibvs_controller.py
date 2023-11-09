import numpy as np
from numpy.linalg import inv
from ibvs_jacobian import ibvs_jacobian

def ibvs_controller(K, pts_des, pts_obs, zs, gain):
    """
    A simple proportional controller for IBVS.

    Implementation of a simple proportional controller for image-based
    visual servoing. The error is the difference between the desired and
    observed image plane points. Note that the number of points, n, may
    be greater than three. The x and y focal lengths in K are guaranteed 
    to be identical.

    Parameters:
    -----------
    K       - 3x3 np.array, camera intrinsic calibration matrix.
    pts_des - 2xn np.array, desired (target) image plane points.
    pts_obs - 2xn np.array, observed (current) image plane points.
    zs      - nx0 np.array, points depth values (may be estimated).
    gain    - Controller gain (lambda).

    Returns:
    --------
    v  - 6x1 np.array, desired tx, ty, tz, wx, wy, wz camera velocities.
    """
    #v = np.zeros((6, 1))

    #--- FILL ME IN ---
    ai=pts_obs[0][0]
    bi=pts_obs[1][0]
    pi=np.array([[ai],[bi]])
    J=ibvs_jacobian(K,pi,zs[0])
    for i in range(1,pts_obs.shape[1]):
        a=pts_obs[0][i]
        b=pts_obs[1][i]
        p=np.array([[a],[b]])
        jt=ibvs_jacobian(K,p,zs[i])
        J=np.concatenate((J,jt),axis=0)

    Jpi=np.matmul(np.linalg.inv(np.matmul(J.transpose(),J)),J.transpose())
    pd=(pts_des-pts_obs)
    pv=np.array([pd.transpose().flatten()]).transpose()
    v=gain*np.matmul(Jpi,pv)
    #print('v= ',v)
    #print('v.shape= ',v.shape)
    #------------------

    correct = isinstance(v, np.ndarray) and \
        v.dtype == np.float64 and v.shape == (6, 1)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return v
