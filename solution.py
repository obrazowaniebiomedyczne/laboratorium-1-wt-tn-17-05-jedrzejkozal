"""
Rozwiązania do laboratorium 1 z Obrazowania Biomedycznego.
"""
import numpy as np

"""
3 - Kwadrat
"""
def square(size, side, start):
    image = np.zeros((size, size)).astype(np.uint8)
    image[start[1]:start[1]+side, start[0]:start[0]+side] = 255
    return image

"""
3 - Koło
"""
def midcircle(size):
    image = np.zeros((size)).astype(np.uint8)
    center = (size[0]/2, size[1]/2)
    radius = min(size)/4
    radius_sqared = radius**2

    for i in range(size[0]):
        for j in range(size[1]):
            if (i-center[0])**2 + (j-center[1])**2 <= radius_sqared:
                image[i, j] = 255

    return image

"""
3 - Szachownica.
"""
def checkerboard(size):
    image = np.zeros((size, size)).astype(np.uint8)
    field_size = size // 8
    for i in range(size):
        for j in range(size):
            if (i // field_size % 2 == 1) != (j // field_size % 2 == 1):
                image[j,i] = 255

    return image

"""
4 - Interpolacja najbliższych sąsiadów.
"""
def calc_coef(old_size, new_size):
    coef_x = old_size[0] / new_size[0]
    coef_y = old_size[1] / new_size[1]
    return coef_x, coef_y

def check_half(arg):
    if (arg - int(arg)) < 0.5:
        return int(arg)
    else:
        return int(arg)+1

def nearest_neighbour_index(x, y, max_index):
    new_x = check_half(x)
    if new_x >= max_index[0]:
        new_x = max_index[0]-1
    new_y = check_half(y)
    if new_y >= max_index[1]:
        new_y = max_index[1]-1
    return (new_x, new_y)

def nn_interpolation(source, new_size):
    source = np.squeeze(source)
    image = np.zeros((new_size)).astype(np.uint8)
    coef_x, coef_y = calc_coef(source.shape, new_size)

    for i in range(new_size[0]):
        for j in range(new_size[1]):
            image[i, j] = source[nearest_neighbour_index(i*coef_x, j*coef_y, source.shape)]
            #image[i, j] = source[int(i*coef_x), int(j*coef_y)]

    return image


"""
5 - Interpolacja dwuliniowa
"""
def calc_coef(old_size, new_size):
    coef_x = old_size[0] / new_size[0]
    coef_y = old_size[1] / new_size[1]
    return coef_x, coef_y

def check_Qx(x):
    Qx = np.zeros((2,2))
    Qx[0, 0] = int(x) #Q11
    Qx[0, 1] = int(x) #Q12
    Qx[1, 0] = int(x)+1 #Q21
    Qx[1, 1] = int(x)+1 #Q22
    return Qx

def check_Qy(y):
    Qy = np.zeros((2,2))
    Qy[0, 0] = int(y) #Q11
    Qy[0, 1] = int(y)+1 #Q12
    Qy[1, 0] = int(y) #Q21
    Qy[1, 1] = int(y)+1 #Q22
    return Qy

def check_Q(x,y):
    return check_Qx(x), check_Qy(y)

def check_fQ(source, Qx, Qy):
    fQ = np.zeros((2,2))

    for i in range(2):
        for j in range(2):
            fQ[i,j] = source[int(Qx[i,j]), int(Qy[i,j])]
    return fQ

def check_x_vector(Qx, x):
    x_vec = np.zeros((1,2))
    x_vec[0,0] = Qx[1,1] - x
    x_vec[0,1] = x - Qx[0,0]
    return x_vec

def check_y_vector(Qy, y):
    y_vec = np.zeros((2,1))
    y_vec[0,0] = Qy[1,1] - y
    y_vec[1,0] = y - Qy[0,0]
    return y_vec

def calc_denominator(Qx, Qy):
    return (Qx[1,1]-Qx[0,0])*(Qy[1,1]-Qy[0,0])

def interpolation(source, x, y):
    Qx, Qy = check_Q(x, y)
    fQ = check_fQ(source, Qx, Qy)

    x_vector = check_x_vector(Qx, x)
    y_vector = check_y_vector(Qy, y)
    denominator = calc_denominator(Qx, Qy)

    A = np.matmul(fQ, y_vector)
    return np.matmul(x_vector, A)/denominator

def bilinear_interpolation(source, new_size):
    source = np.squeeze(source)
    image = np.zeros((new_size)).astype(np.uint8)
    coef_x, coef_y = calc_coef(source.shape, new_size)

    for i in range(new_size[0]):
        for j in range(new_size[1]):
            image[i, j] = interpolation(source, i*coef_x, j*coef_y)

    return image
