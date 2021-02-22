"""

copied from my cpeptools package
"""
import numpy as np
from numpy.linalg import eig, inv
from scipy.optimize import fsolve
from scipy.spatial.distance import cdist

from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

"""
#TODO:
    - see to remove sklearn dependency (code below have right eigenvals not right eigenvectors)
import numpy as np
from sklearn import decomposition
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
X = X[:, :-1]
X.shape
pca = decomposition.PCA(n_components=3)
pca.fit(X)
pca.transform(X)
pca.explained_variance_


x_std = X

cov = np.cov(x_std.T)
ev , eig = np.linalg.eig(cov)
a = eig.dot(x_std.T)
ev
a.T

https://stats.stackexchange.com/questions/235882/pca-in-numpy-and-sklearn-produces-different-results
--------------------


Only deals with 2D/3D points, no molecule


Ellipse:
- https://math.stackexchange.com/questions/2349022/is-it-possible-to-find-the-distance-between-two-points-on-the-circumference-of-a
- https://www.mathsisfun.com/geometry/ellipse-perimeter.html
- https://blender.stackexchange.com/questions/60562/ellipse-construction-advice/60624#60624
- http://mathworld.wolfram.com/Ellipse.html
"""

def center_2D_points(x,y):
    center_x, center_y = (sum(x) / len(x), sum(y) / len(y))
    print(center_x, center_y)
    return x-center_x, y-center_y

def get_convex_hull(coords, dim = 2, needs_at_least_n_points = 6): #FIXME restrict only for 2D?
    """
    For fitting an ellipse, at least 6 points are needed


    Parameters
    ----------
    coords : 2D np.array of points
    dim : dimensions to keep when calculating convex hull

    Returns
    ---------
    coords_hull : 2D np.array of points
            keeps original number of dimension as input coords
    """
    assert len(coords[0]) >= dim

    hull = ConvexHull([i[:dim] for i in coords])

    coords_hull = [coords[i] for i in range(len(coords)) if i in hull.vertices]

    for i in range(needs_at_least_n_points - len(hull.vertices)):
        coords_hull.append(0.9999 * coords_hull[i]) #making the point slightly different

    coords_hull = np.array(coords_hull)

    return coords_hull

def get_pca(coords):
    """
    Parameters
    -----------
    coords : 2D np.array of points

    Returns
    ---------
    new_coords : 2D np.array of points
            keeps original number of dimension as input coords

    variance_ratio : tuple


    """
    pca = PCA(n_components=3)
    # pca.fit(coords)
    # new_coords = pca.transform(coords)
    new_coords = pca.fit_transform(coords)

    return  new_coords, pca.explained_variance_ratio_


######### related to ellipse geometry

def calculate_ellipse_radii(guess, eccentricity = 0, perimeter = 2 * np.pi*1):
    """
    returns a,b where a <= b

    """
    return fsolve(ellipse_radii_test, guess, args = (eccentricity, perimeter))

def ellipse_radii_test(radii, eccentricity = 0, perimeter = 2*np.pi*1):
    """
        simultaneous equations to be solved invloving ellipse radii and the custom input eccentricty, or perimeter
    """
    a,b = radii

    return (np.sqrt(np.absolute(1 - (b**2)/(a**2))) - eccentricity,
            # perimeter approximation from https://www.mathsisfun.com/geometry/ellipse-perimeter.html
            np.pi * (3 * (a + b) - np.sqrt(np.absolute((3 * a + b) * (a + 3 * b)))) - perimeter)

def get_points_on_ellipse(a, b, numPoints, bond_length_list = None, startAngle = 0, verbose = False, increment = 0.01):
    """
        Currently only works for ellipse centered on origin
        the points are drawn from the +ve x axis in the order of the quardrants

        Paramters:
        ----------------
        startAngle :  float
            the angle the first point makes with the axis, default is 0 (i.e) first point on the x-axis

        one of `numPoints` and `bond_length_list` needs to be None
        ----------------
    """
    def distance(x1,y1,x2,y2):
        return np.sqrt((x2-x1)**2 + (y2-y1)**2)

    # if numPoints is None and bond_length_list is None:
    #     print("both cannot be None")
    #     return

    x0 = a
    y0 = 0
    angle = 0
    d = 0
    while(angle <= 360):
        x = a * np.cos(np.radians(angle))
        y = b * np.sin(np.radians(angle))
        d += distance(x0,y0,x,y)
        x0 = x
        y0 = y
        angle += increment
    if verbose:
        print("The estimated circumference of ellipse is {:f}".format(d))

    if bond_length_list is not None and len(bond_length_list) != numPoints:
        print("number of atoms do not equal {} compared to {}".format(len(bond_length_list), numPoints))
        return
    if bond_length_list is not None and not np.isclose(sum(bond_length_list), d, rtol = 0, atol = 0.0001): #0.01 pm accuracy
        print("distance do no agree {} compared to {}".format(sum(bond_length_list), d))
        return

    points = []

    if bond_length_list is None:
        arcLength = d/numPoints
        angle = 0
        x0 = a
        y0 = 0
        angle0 = 0
        while(angle0 < startAngle):
            angle += increment
            x = a * np.cos(np.radians(angle))
            y = b * np.sin(np.radians(angle))
            x0 = x
            y0 = y
            angle0 = angle
        for i in range(numPoints):
            dist = 0
            while(dist < arcLength):
                angle += increment
                x = a * np.cos(np.radians(angle))
                y = b * np.sin(np.radians(angle))
                dist += distance(x0,y0,x,y)
                x0 = x
                y0 = y
            if verbose:
                print(
                    "{} : angle = {:.2f}\tdifference = {:.2f}\tDistance {:.2f}"
                    .format(i+1,angle, angle-angle0,dist))
            points.append([x0, y0])
            angle0 = angle
        return np.array(points)

    elif bond_length_list is not None:
        counter = 0
        angle = 0
        x0 = a
        y0 = 0
        angle0 = 0
        while(angle0 < startAngle):
            angle += increment
            x = a * np.cos(np.radians(angle))
            y = b * np.sin(np.radians(angle))
            x0 = x
            y0 = y
            angle0 = angle

        for idx,arcLength in enumerate(bond_length_list):
            dist = 0
            while(dist < arcLength):
                angle += increment
                x = a * np.cos(np.radians(angle))
                y = b * np.sin(np.radians(angle))
                dist += distance(x0,y0,x,y)
                x0 = x
                y0 = y
            if verbose:
                print(
                    "{} : angle = {:.2f}\tdifference = {:.2f}\tDistance {:.2f}"
                    .format(i+1,angle, angle-angle0,dist))
            points.append([x0, y0])
            angle0 = angle
        return np.array(points)



def fit_ellipse(x,y):
    """
    Parameters
    ----------
    x : np.array
        x-coordinates of points
    y : np.array
        y-coordinates of points

    Returns
    ----------
    fitted_ellipse_obj
    """
    x = x[:,np.newaxis] - np.mean(x)
    y = y[:,np.newaxis] - np.mean(y)
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))

    #############################
    # fixing using answer in : https://stackoverflow.com/questions/39693869/fitting-an-ellipse-to-a-set-of-data-points-in-python
    # n = np.argmax(np.abs(E))
    n = np.argmax(E)
    #############################

    fitted_ellipse_obj = V[:,n]
    return fitted_ellipse_obj

def get_eccentricity(fitted_ellipse_obj):
    """
    fitted_ellipse_obj is a list of value required to determine the ellipse property
    """
    a,b = ellipse_axis_length(fitted_ellipse_obj)
    #FIXME: this is hack as it is possible when fitting to
    # CH, the number of points are small and the ellipse is not so great
    # somehow semiminor axis can be bigger than semimajor axis,
    # instead of converted `1 - b**2 / a ** 2)` -> `np.absolute(1 - b**2 / a ** 2))`
    # this gives more drastic difference for Eccen and Eccen_CH compared to
    # just swapping the order of a and b
    # many frames in simulation gave b > a for CH , so I swap them here

    if b > a:
        # print("semimajor axis is bigger than semiminor", a,b)
        a,b = b,a
    eccen = np.sqrt(1 - b**2 / a ** 2)
    return eccen #number (0,1)

def ellipse_center(a):
    """
    Parameters
    ----------
    a : fitted_ellipse_obj
    """
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])

def ellipse_angle_of_rotation( a ):
    """
    Parameters
    ----------
    a : fitted_ellipse_obj
    """
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return 0.5*np.arctan(2*b/(a-c))


def ellipse_axis_length( a ):
    """
    Parameters
    ----------
    a : fitted_ellipse_obj

    Returns
    ----------
    ellipse radii
    """
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    #FIXME sometimes down1 or down2 is negative, here I have chagned to use the absolute value, I DO NOT YET KNOW this is valid or not.
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)

    return np.array([res1, res2]) #semi major and semi minor axis, the order is not necessarily correct

def ellipse_angle_of_rotation2( a ):
    """
    Parameters
    ----------
    a : fitted_ellipse_obj
    """
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi/2
    else:
        if a > c:
            return np.arctan(2*b/(a-c))/2
        else:
            return np.pi/2 + np.arctan(2*b/(a-c))/2


def get_info_from_e_obj(e_obj, func_list = [ellipse_axis_length, ellipse_angle_of_rotation, ellipse_center]):
    for f in func_list:
        yield f(e_obj)

def place_points_on_ellipse(semimaj, semimin, phi, x_cent, y_cent, theta_num = 1e3):
    # Generate data for ellipse structure
    theta = np.linspace(0,2*np.pi,theta_num)
    r = 1 / np.sqrt((np.cos(theta))**2 + (np.sin(theta))**2)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    data = np.array([x,y])
    S = np.array([[semimaj,0],[0,semimin]])
    R = np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])
    T = np.dot(R,S)
    data = np.dot(T,data)
    data[0] += x_cent
    data[1] += y_cent

    return data

def plot_ellipse(semimaj=1,semimin=1,phi=0,x_cent=0,y_cent=0,theta_num=1e3,ax=None,plot_kwargs=None,\
                    fill=False,fill_kwargs=None,data_out=False,cov=None,mass_level=0.68):
    import matplotlib.pyplot as plt
    from scipy.stats import chi2
    '''
        An easy to use function for plotting ellipses in Python 2.7!

        The function creates a 2D ellipse in polar coordinates then transforms to cartesian coordinates.
        It can take a covariance matrix and plot contours from it.

        semimaj : float
            length of semimajor axis (always taken to be some phi (-90<phi<90 deg) from positive x-axis!)

        semimin : float
            length of semiminor axis

        phi : float
            angle in radians of semimajor axis above positive x axis

        x_cent : float
            X coordinate center

        y_cent : float
            Y coordinate center

        theta_num : int
            Number of points to sample along ellipse from 0-2pi

        ax : matplotlib axis property
            A pre-created matplotlib axis

        plot_kwargs : dictionary
            matplotlib.plot() keyword arguments

        fill : bool
            A flag to fill the inside of the ellipse

        fill_kwargs : dictionary
            Keyword arguments for matplotlib.fill()

        data_out : bool
            A flag to return the ellipse samples without plotting

        cov : ndarray of shape (2,2)
            A 2x2 covariance matrix, if given this will overwrite semimaj, semimin and phi

        mass_level : float
            if supplied cov, mass_level is the contour defining fractional probability mass enclosed
            for example: mass_level = 0.68 is the standard 68% mass

    '''
    # Get Ellipse Properties from cov matrix
    if cov is not None:
        eig_vec,eig_val,u = np.linalg.svd(cov)
        # Make sure 0th eigenvector has positive x-coordinate
        if eig_vec[0][0] < 0:
            eig_vec[0] *= -1
        semimaj = np.sqrt(eig_val[0])
        semimin = np.sqrt(eig_val[1])
        if mass_level is None:
            multiplier = np.sqrt(2.279)
        else:
            distances = np.linspace(0,20,20001)
            chi2_cdf = chi2.cdf(distances,df=2)
            multiplier = np.sqrt(distances[np.where(np.abs(chi2_cdf-mass_level)==np.abs(chi2_cdf-mass_level).min())[0][0]])
        semimaj *= multiplier
        semimin *= multiplier
        phi = np.arccos(np.dot(eig_vec[0],np.array([1,0])))
        if eig_vec[0][1] < 0 and phi > 0:
            phi *= -1


    data = place_points_on_ellipse(semimaj, semimin, phi, x_cent, y_cent, theta_num)

    # Output data?
    if data_out == True:
        return data

    # Plot!
    return_fig = False
    if ax is None:
        return_fig = True
        fig,ax = plt.subplots()

    if plot_kwargs is None:
        ax.plot(data[0],data[1],color='b',linestyle='-')
    else:
        ax.plot(data[0],data[1],**plot_kwargs)

    if fill == True:
        ax.fill(data[0],data[1],**fill_kwargs)
    if return_fig == True:
        return fig