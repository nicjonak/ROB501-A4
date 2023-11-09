import numpy as np
from numpy.linalg import inv
from ibvs_jacobian import ibvs_jacobian

def ibvs_depth_finder(K, pts_obs, pts_prev, zs_guess, v_cam):
    """
    Compute estimated 

    The function computes the Jacobian for image-based visual servoing,
    given the camera matrix K, an image plane point, and the estimated
    depth of the point. The x and y focal lengths in K are guaranteed 
    to be identical.

    Parameters:
    -----------
    K        - 3x3 np.array, camera intrinsic calibration matrix.
    pts_obs  - 2xn np.array, observed (current) image plane points.
    pts_prev - 2xn np.array, observed (previous) image plane points.
    zs_guess - nx0 np.array, points depth values (estimated guess).
    v_cam    - 6x1 np.array, camera velocity (last commmanded).

    Returns:
    --------
    zs_est - nx0 np.array, updated, estimated depth values for each point.
    """
    n = pts_obs.shape[1]
    #J = np.zeros((2*n, 6))
    #zs_est = np.zeros(n)

    #--- FILL ME IN ---
    
    pvt=pts_obs-pts_prev
    pv=np.array([pvt.transpose().flatten()]).transpose()
    v=np.split(v_cam,2)[0]
    w=np.split(v_cam,2)[1]

    ai=pts_obs[0][0]
    bi=pts_obs[1][0]
    pi=np.array([[ai],[bi]])
    J=ibvs_jacobian(K,pi,zs_guess[0])
    for i in range(1,n):
       ap=pts_obs[0][i]
       bp=pts_obs[1][i]
       p=np.array([[ap],[bp]])
       jtemp=ibvs_jacobian(K,p,zs_guess[i])
       J=np.concatenate((J,jtemp),axis=0)

    Jt=np.hsplit(J,2)[0]
    Jw=np.hsplit(J,2)[1]
    
    A=np.matmul(Jt,v)
    #print('A= ',A)
    b=pv-np.matmul(Jw,w)
    #print('b= ',b)
    ze=[]
    for i in range(0,2*n,2):
        at=np.array([[A[i][0]],[A[i+1][0]]])
        bt=np.array([[b[i][0]],[b[i+1][0]]])
        zet=np.linalg.lstsq(at,bt,rcond=None)[0][0][0]
        ze=ze+[1.0/zet]
    
    #print('ze= ',ze)
    zs_est=np.array(ze)
    #------------------

    correct = isinstance(zs_est, np.ndarray) and \
        zs_est.dtype == np.float64 and zs_est.shape == (n,)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return zs_est
