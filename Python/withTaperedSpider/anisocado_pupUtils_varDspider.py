# Alterations to the anisocado code by Jani Achren, Incident Angle Oy
# <jani.achren@incidentangle.fi>

import numpy as np

def generateEeltPupilReflectivity(refl, npt, dspider, i0, j0, pixscale, gap,
                                  rotdegree, D=40.0, softGap=False):
    """
    Generates a map of the reflectivity of the EELT pupil, on an array
    of size (npt, npt).

    :returns: pupil image (npt, npt), with the same type of input argument refl
    :param float/int/bool refl: scalar value or 1D-array of the reflectivity of
           the segments.
           If refl is scalar, the value will be replicated for all segments.
           If refl is a 1D array, then it shall contain the reflectivities
           of all segments.
           On output, the data type of the pupil map will be the same as refl.
    :param int npt: size of the output array
    :param float dspider: width of spiders in meters
    :param float i0, j0: index of pixels where the pupil should be centred.
                         Can be floating-point indexes.
    :param float pixscale: size of a pixel of the image, in meters.
    :param float gap: half-space between segments in meters
    :param float rotdegree: rotation angle of the pupil, in degrees.
    :param float D: diameter of the pupil. For the nominal EELT, D shall
                    be set to 40.0
    :param bool softGap: if False, the gap between segments is binary 0/1
          depending if the pixel is within the gap or not. If True, the gap
          is a smooth region of a fwhm of 2 pixels with a depth related to the
          gap width.
    """
    rot = rotdegree * np.pi / 180

    # Generation of segments coordinates.
    # hx and hy have a shape [6,798] describing the 6 vertex of the 798
    # hexagonal mirrors
    hx, hy = generateCoordSegments(D, rot)

    # From the data of hex mirrors, we build the pupil image according
    # to the properties defined by input argument <refl>
    pup = generateSegmentProperties(refl, hx, hy, i0, j0, pixscale, gap, npt, D,
                                    softGap=softGap)

    # SPIDERS ............................................
    nspider = 3  # for the day where we have more/less spiders ;-)
       
    if (dspider > 0 and nspider > 0):
        pup = pup * fillSpider(npt, nspider, dspider, i0, j0, pixscale, rot, spider_thickness_func=taper)

    return pup

def taper(r, R_inner=5.55, R_outer=19.2):
    """
    ELT M2 spider bar taper profile.

    Parameters
    ----------
    r : ndarray
        Radius array (meters).
    R_inner : float
        Radius of central obscuration (meters).
    R_outer : float
        Outer pupil radius (meters).

    Returns
    -------
    thickness : ndarray
        Spider thickness at each radius (meters).
    """
    thickness = np.zeros_like(r)

    # Region 1: taper down from 540 mm to 310 mm over 0.5 m
    mask1 = (r >= R_inner) & (r < R_inner + 0.5)
    thickness[mask1] = 0.54 - (0.23/0.5)*(r[mask1] - R_inner)

    # Region 2: constant 310 mm
    mask2 = (r >= R_inner + 0.5) & (r < R_outer - 0.5)
    thickness[mask2] = 0.31

    # Region 3: taper back up to 540 mm over last 0.5 m
    mask3 = (r >= R_outer - 0.5) & (r <= R_outer)
    thickness[mask3] = 0.31 + (0.23/0.5)*(r[mask3] - (R_outer - 0.5))

    return thickness



def generateCoordSegments(D, rot):
    """
    Computes the coordinates of the corners of all the hexagonal
    segments of M1.
    Result is a tuple of arrays(6, 798).

    :param float D: D is the pupil diameter in meters, it must be set to 40.0 m
    for the nominal EELT.
    :param float rot: pupil rotation angle in radians

    """
    V3 = np.sqrt(3)
    pitch = 1.227314  # no correction du bol
    pitch = 1.244683637214  # diametre du cerle INSCRIT
    # diamseg = pitch*2/V3  # diametre du cercle contenant TOUT le segment
    # print("segment diameter : %.6f\n" % diamseg)

    # Creation d'un pattern hexa avec pointes selon la variable <ly>
    lx, ly = createHexaPattern(pitch, 35 * pitch)
    ll = np.sqrt(lx ** 2 + ly ** 2)
    # Elimination des segments non valides grace a 2 nombres parfaitement
    # empiriques ajustes a-la-mano.
    inner_rad, outer_rad = 4.1, 15.4  # nominal, 798 segments
    nn = (ll > inner_rad * pitch) & (ll < outer_rad * pitch);
    lx = lx[nn]
    ly = ly[nn]
    lx, ly = reorganizeSegmentsOrderESO(lx, ly)
    ll = np.sqrt(lx ** 2 + ly ** 2)

    # n = ll.shape[0]
    # print("Nbre de segments : %d\n" % n)
    # Creation d'un hexagone-segment avec pointe dirigee vers
    # variable <hx> (d'ou le cos() sur hx)
    th = np.linspace(0, 2 * np.pi, 7)[0:6]
    hx = np.cos(th) * pitch / V3
    hy = np.sin(th) * pitch / V3

    # Le maillage qui permet d'empiler des hexagones avec sommets 3h-9h
    # est un maillage hexagonal avec sommets 12h-6h, donc a 90°.
    # C'est pour ca qu'il a fallu croiser les choses avant.
    x = (lx[None, :] + hx[:, None])
    y = (ly[None, :] + hy[:, None])
    r = np.sqrt(x ** 2 + y ** 2)
    R = 95.7853
    rrc = R / r * np.arctan(r / R)  # correction factor
    x *= rrc
    y *= rrc

    nominalD = 40.0  # size of the OFFICIAL E-ELT
    if D != nominalD:
        x *= D / nominalD
        y *= D / nominalD

    # Rotation matrices
    mrot = np.array([[np.cos(rot), np.sin(rot)], [-np.sin(rot), np.cos(rot)]])

    # rotation of coordinates
    # le tableau [x,y] est de taille (2,6,798). Faut un transpose a la con
    # pour le transformer en (6,2,798) pour pouvoir faire le np.dot
    # correctement. En sortie, xrot est (2,6,798).
    xyrot = np.dot(mrot, np.transpose(np.array([x, y]), (1, 0, 2)))

    return xyrot[0], xyrot[1]


def generateSegmentProperties(attribute, hx, hy, i0, j0, scale, gap, N, D,
                              softGap=0):
    """
    Builds a 2D image of the pupil with some attributes for each of the
    segments. Those segments are described from arguments hx and hy, that
    are produced by the function generateCoordSegments(D, rot).

    When attribute is a phase, then it must be a float array of dimension
    [3, 798] with the dimension 3 being piston, tip, and tilt.
    Units of phase is xxx rms, and the output of the procedure will be
    in units of xxx.


    :returns: pupil image (N, N), with the same type of input argument attribute

    :param float/int/bool attribute: scalar value or 1D-array of the
        reflectivity of the segments or 2D array of phase
           If attribute is scalar, the value will be replicated for all segments
           If attribute is a 1D array, then it shall contain the reflectivities
           of all segments.
           If attribute is a 2D array then it shall contain the piston, tip
           and tilt of the segments. The array shall be of dimension
           [3, 798] that contains [piston, tip, tilt]
           On output, the data type of the pupil map will be the same as input
    :param float hx, hy: arrays [6,:] describing the segment shapes. They are
        generated using generateCoordSegments()
    :param float dspider: width of spiders in meters
    :param float i0, j0: index of pixels where the pupil should be centred.
                         Can be floating-point indexes.
    :param float scale: size of a pixel of the image, in meters.
    :param float gap: half-space between segments in meters
    :param int N: size of the output array (N,N)
    :param float D: diameter of the pupil. For the nominal EELT, D shall
                    be set to 40.0
    :param bool softGap: if False, the gap between segments is binary 0/1
          depending if the pixel is within the gap or not. If True, the gap
          is a smooth region of a fwhm of 2 pixels with a depth related to the
          gap width.



    """

    # number of segments
    nseg = hx.shape[-1]
    # If <attribute> is a scalar, then we make a list. It will be required
    # later on to set the attribute to each segment.
    if np.isscalar(attribute):
        attribute = np.array([attribute] * nseg)

    # the pupil map is created with the same data type as <attribute>
    pupil = np.zeros((N, N), dtype=getdatatype(attribute))

    # average coord of segments
    x0 = np.mean(hx, axis=0)
    y0 = np.mean(hy, axis=0)
    # avg coord of segments in pixel indexes
    x0 = x0 / scale + i0
    y0 = y0 / scale + j0
    # size of mini-support
    hexrad = 0.75 * D / 40. / scale
    ix0 = np.floor(x0 - hexrad).astype(int) - 1
    iy0 = np.floor(y0 - hexrad).astype(int) - 1
    segdiam = np.ceil(hexrad * 2 + 1).astype(int) + 1

    n = attribute.shape[0]
    if n != 3:
        # attribute is a signel value : either reflectivity, or boolean,
        # or just piston.
        if softGap != 0:
            # Soft gaps
            # The impact of gaps are modelled using a simple function:
            # Lorentz, 1/(1+x**2)
            # The fwhm is always equal to 2 pixels because the gap is supposed
            # to be "small/invisible/undersampled". The only visible thing is
            # the width of the impulse response, chosen 2-pixel wide to be
            # well sampled.
            # The "depth" is related to the gap width. The integral of a
            # Lorentzian of 2 pix wide is PI. Integral of a gap of width 'gap'
            # in pixels is 'gap'.
            # So the depth equals to gap/scale/np.pi.
            for i in range(nseg):
                indx, indy, distedge = fillPolygon(hx[:, i], hy[:, i],
                                                   i0 - ix0[i], j0 - iy0[i],
                                                   scale, gap * 0., segdiam,
                                                   index=1)
                pupil[indx + ix0[i], indy + iy0[i]] = attribute[i] * (
                            1. - (gap / scale / np.pi) / (
                                1 + (distedge / scale) ** 2))
        else:
            # Hard gaps
            for i in range(nseg):
                indx, indy, distedge = fillPolygon(hx[:, i], hy[:, i],
                                                   i0 - ix0[i], j0 - iy0[i],
                                                   scale, gap, segdiam, index=1)
                pupil[indx + ix0[i], indy + iy0[i]] = attribute[i]
    else:
        # attribute is [piston, tip, tilt]
        minimap = np.zeros((segdiam, segdiam))
        xmap = np.arange(segdiam) - segdiam / 2
        xmap, ymap = np.meshgrid(xmap, xmap, indexing='ij')  # [x,y] convention
        pitch = 1.244683637214  # diameter of inscribed circle
        diamseg = pitch * 2 / np.sqrt(3)  # diameter of circumscribed circle
        diamfrizou = (pitch + diamseg) / 2 * D / 40.  # average diameter
        # Calcul du facteur de mise a l'echelle pour l'unite des tilts.
        # xmap et ymap sont calculees avec un increment de +1 pour deux pixels
        # voisins, donc le facteur a appliquer est tel que l'angle se conserve
        # donc factunit*1 / scale = 4*factunit
        factunit = 4 * scale / diamfrizou
        for i in range(nseg):
            indx, indy, _ = fillPolygon(hx[:, i], hy[:, i], i0 - ix0[i],
                                        j0 - iy0[i], scale, 0., segdiam,
                                        index=1)
            minimap = attribute[0, i] + (factunit * attribute[1, i]) * xmap + (
                        factunit * attribute[2, i]) * ymap
            pupil[indx + ix0[i], indy + iy0[i]] = minimap[indx, indy]

    return pupil


#def fillSpider(N, nspider, dspider, i0, j0, scale, rot, spider_thickness_func=None):
#    """
#    Creates a boolean spider mask on a map of dimensions (N,N)
#    The spider is centred at floating-point coords (i0,j0).
#
#    :returns: spider image (boolean)
#    :param int N: size of output image
#    :param int nspider: number of spiders
#    :param float dspider: width of spiders
#    :param float i0: coord of spiders symmetry centre
#    :param float j0: coord of spiders symmetry centre
#    :param float scale: size of a pixel in same unit as dspider
#    :param float rot: rotation angle in radians
#
#    """
#    a = np.ones((N, N), dtype=np.bool_)
#    X = (np.arange(N) - i0) * scale
#    Y = (np.arange(N) - j0) * scale
#    X, Y = np.meshgrid(X, Y, indexing='ij')  # convention d'appel [x,y]
#    w = 2 * np.pi / nspider
#    # rot += np.pi/2  # parce que c'est comme ca !!
#    for i in range(nspider):
#        nn = (abs(X * np.cos(i * w - rot) + Y * np.sin(i * w - rot)) < dspider / 2.)
#        a[nn] = False
#    return a
    
def fillSpider(N, nspider, dspider, i0, j0, scale, rot, spider_thickness_func=None): #JA add. (varDspider)
    """
    Creates a boolean spider mask on a map of dimensions (N,N).
    The spider is centred at floating-point coords (i0,j0).

    Parameters
    ----------
    N : int
        Size of output image
    nspider : int
        Number of spiders
    dspider : float
        Constant width of spiders (used if spider_thickness_func is None)
    i0, j0 : float
        Centre coordinates
    scale : float
        Pixel size in same unit as dspider
    rot : float
        Rotation angle in radians
    spider_thickness_func : callable or None
        Optional function thickness(r) -> float, where r is radius from centre.
        If None, constant dspider is used.
    """
    a = np.ones((N, N), dtype=np.bool_)
    X = (np.arange(N) - i0) * scale
    Y = (np.arange(N) - j0) * scale
    X, Y = np.meshgrid(X, Y, indexing='ij')
    r = np.sqrt(X**2 + Y**2)
    w = 2 * np.pi / nspider

    for i in range(nspider):
        # thickness is now actually used in the condition
        if spider_thickness_func is None:
            thickness = dspider
        else:
            thickness = spider_thickness_func(r)

        nn = (np.abs(X * np.cos(i * w - rot) + Y * np.sin(i * w - rot)) < thickness / 2.)
        a[nn] = False
    return a
    
    
def createHexaPattern(pitch, supportSize):
    """
    Cree une liste de coordonnees qui decrit un maillage hexagonal.
    Retourne un tuple (x,y).

    Le maillage est centre sur 0, l'un des points est (0,0).
    Une des pointes de l'hexagone est dirigee selon l'axe Y, au sens ou le
    tuple de sortie est (x,y).

    :param float pitch: distance between 2 neighbour points
    :param int supportSize: size of the support that need to be populated

    """
    V3 = np.sqrt(3)
    nx = int(np.ceil((supportSize / 2.0) / pitch) + 1)
    x = pitch * (np.arange(2 * nx + 1) - nx)
    ny = int(np.ceil((supportSize / 2.0) / pitch / V3) + 1)
    y = (V3 * pitch) * (np.arange(2 * ny + 1) - ny)
    x, y = np.meshgrid(x, y, indexing='ij')
    x = x.flatten()
    y = y.flatten()
    peak_axis = np.append(x, x + pitch / 2.)  # axe dirige selon sommet
    flat_axis = np.append(y, y + pitch * V3 / 2.)  # axe dirige selon plat
    return flat_axis, peak_axis


def reorganizeSegmentsOrderESO(x, y):
    """
    Reorganisation des segments facon ESO.
    Voir
    ESO-193058 Standard Coordinate System and Basic Conventions

    :param float x: tableau des centres X des segments
    :param float y: idem Y
    :return tuple (x,y): meme tuple que les arguments d'entree, mais tries.

    """
    # pi/2, pi/6, 2.pi, ...
    pi_3 = np.pi / 3
    pi_6 = np.pi / 6
    pix2 = 2 * np.pi
    # calcul des angles
    t = (np.arctan2(y, x) + pi_6 - 1e-3) % (pix2)
    X = np.array([])
    Y = np.array([])
    A = 100.
    for k in range(6):
        sector = (t > k * pi_3) & (t < (k + 1) * pi_3)
        u = k * pi_3
        distance = (A * np.cos(u) - np.sin(u)) * x[sector] + (
                    np.cos(u) + A * np.sin(u)) * y[sector]
        indsort = np.argsort(distance)
        X = np.append(X, x[sector][indsort])
        Y = np.append(Y, y[sector][indsort])
    return X, Y


def getdatatype(truc):
    """
    Returns the data type of a numpy variable, either scalar value or array.

    """
    if np.isscalar(truc):
        return type(truc)
    else:
        return type(truc.flatten()[0])


def fillPolygon(x, y, i0, j0, scale, gap, N, index=0):
    """
    From a list of points defined by their 2 coordinates list
    x and y, creates a filled polygon with sides joining the points.
    The polygon is created in an image of size (N, N).
    The origin (x,y)=(0,0) is mapped at pixel i0, j0 (both can be
    floating-point values).
    Arrays x and y are supposed to be in unit U, and scale is the
    pixel size in U units.

    :returns: filled polygon (N, N), boolean
    :param float x, y: list of points defining the polygon
    :param float i0, j0: index of pixels where the pupil should be centred.
                         Can be floating-point indexes.
    :param float scale: size of a pixel of the image, in same unit as x and y.
    :param float N: size of output image.

    :Example:
    x = np.array([1,-1,-1.5,0,1.1])
    y = np.array([1,1.5,-0.2,-2,0])
    N = 200
    i0 = N/2
    j0 = N/2
    gap = 0.
    scale = 0.03
    pol = fillPolygon(x, y, i0, j0, scale, gap, N, index=2)

    """
    # define coordinates map centred on (i0,j0) with same units as x,y.
    X = (np.arange(N) - i0) * scale
    Y = (np.arange(N) - j0) * scale
    X, Y = np.meshgrid(X, Y, indexing='ij')  # indexage [x,y]

    # define centre of polygon x0, y0
    x0 = np.mean(x)
    y0 = np.mean(y)

    # compute angles of all pixels coordinates of the map, and all
    # corners of the polygon
    T = (np.arctan2(Y - y0, X - x0) + 2 * np.pi) % (2 * np.pi)
    t = (np.arctan2(y - y0, x - x0) + 2 * np.pi) % (2 * np.pi)

    # on va voir dans quel sens ca tourne. Je rajoute ca pour que ca marche
    # quel que soit le sens de rotation des points du polygone.
    # En fait, j'aurais peut etre pu classer les points par leur angle, pour
    # etre sur que ca marche meme si les points sont donnes dans ts les cas
    sens = np.median(np.diff(t))
    if sens < 0:
        x = x[::-1]
        y = y[::-1]
        t = t[::-1]

    # re-organise order of polygon points so that it starts from
    # angle = 0, or at least closest to 0.
    imin = t.argmin()  # position of the minimum
    if imin != 0:
        x = np.roll(x, -imin)
        y = np.roll(y, -imin)
        t = np.roll(t, -imin)

    # For each couple of consecutive corners A, B, of the polygon, one fills
    # the triangle AOB with True.
    # Last triangle has a special treatment because it crosses the axis
    # with theta=0=2pi
    n = x.shape[0]  # number of corners of polygon
    indx, indy = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    distedge = np.array([], dtype=np.float64)
    for i in range(n):
        j = i + 1  # j=element next i except when i==n : then j=0 (cycling)
        if j == n:
            j = 0
            sub = np.where((T >= t[-1]) | (T <= (t[0])))
        else:
            sub = np.where((T >= t[i]) & (T <= t[j]))
        # compute unitary vector des 2 sommets
        dy = y[j] - y[i]
        dx = x[j] - x[i]
        vnorm = np.sqrt(dx ** 2 + dy ** 2)
        dx /= vnorm
        dy /= vnorm
        # calcul du produit vectoriel
        crossprod = dx * (Y[sub] - y[i]) - dy * (X[sub] - x[i])
        tmp = crossprod > gap
        indx = np.append(indx, sub[0][tmp])
        indy = np.append(indy, sub[1][tmp])
        distedge = np.append(distedge, crossprod[tmp])

    # choice of what is returned : either only the indexes, or the
    # boolean map
    if index == 1:
        return (indx, indy, distedge)
    elif index == 2:
        a = np.zeros((N, N))
        a[indx, indy] = distedge
        return a
    else:
        a = np.zeros((N, N), dtype=np.bool_)
        a[indx, indy] = True  # convention [x,y]

    return a

