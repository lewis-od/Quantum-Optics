import os
import shutil
import argparse
import numpy as np
import qutip as qu
import matplotlib.pyplot as plt

VERBOSE = False

def rand_complex(modulus):
    r = np.random.rand() * modulus
    theta = np.random.rand() * 2 * np.pi
    return r * np.exp(1j*theta)

def cat(T, alpha, theta=0):
    a = qu.coherent(T, alpha)
    b = qu.coherent(T, -alpha)
    return (a + np.exp(1j*theta)*b).unit()

def squeezed(T, z):
    S = qu.squeeze(T, z)
    vac = qu.basis(T, 0)
    return S * vac

def gen_wigner(n, T, xvec):
    types = ['fock', 'coherent', 'squeezed', 'cat']
    wigners = np.empty([n, 200, 200])
    labels = np.empty(n)
    params = np.empty(n, dtype=np.complex64)

    state = None
    for i in range(n):
        label = i % len(types)
        if label == 0:
            n_photons = np.random.randint(0, T)
            params[i] = n_photons
            state = qu.basis(T, n_photons)
        elif label == 1:
            alpha = rand_complex(5)
            params[i] = alpha
            state = qu.coherent(T, alpha)
        elif label == 2:
            z = rand_complex(1.4)
            params[i] = z
            state = squeezed(T, z)
        elif label == 3:
            alpha = rand_complex(1.5)
            params[i] = alpha
            # theta = np.random.rand() * 2 * np.pi
            state = cat(T, alpha)
        data = qu.wigner(state, xvec, xvec)
        wigners[i] = data
        labels[i] = label
        if VERBOSE:
            print("Wigner function number {} calculated.".format(i))

    return wigners, labels, params

def save_data(folder, wigners, xvec, labels, params, imsize, dpi):
    # Remove folder if it exists
    try:
        cur_dir = os.path.abspath(os.path.join(__file__, os.pardir))
    except:
        cur_dir = os.path.abspath(".")
    img_dir = os.path.join(cur_dir, folder)
    if os.path.isdir(img_dir):
        shutil.rmtree(img_dir)
    # Create the folder
    os.makedirs(img_dir)

    fig = plt.figure(figsize=(imsize/dpi, imsize/dpi), dpi=dpi)
    # fig = plt.figure()
    ax = plt.axes([0,0,1,1])
    plt.axis('off')
    for n in range(len(wigners)):
        data = wigners[n]
        # Plot the Wigner function
        w_map = qu.wigner_cmap(data)
        plt.imshow(data, cmap=w_map, interpolation='bilinear', aspect='equal')

        # Save the image
        fname = "wigner_{}.png".format(n)
        path = os.path.join(img_dir, fname)
        plt.savefig(path, bbox_inches=0.0, dpi=dpi)
        if VERBOSE:
            print("Image {} saved.".format(n))
    # Save the labels as a numpy array
    np.save(os.path.join(img_dir, "labels"), labels)
    # Save the actual Wigner function values, the axis values, and the value
    # of the parameter (e.g n, z, alpha, etc) of each state
    np.savez(os.path.join(img_dir, "raw"),
        data=wigners, params=params, axes=xvec)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate images of the Wigner function of states")
    parser.add_argument('--training', type=int, required=False, default=10,
        help="Number of images to generate for the training dataset",
        metavar='NUM')
    parser.add_argument('--xlim', type=float, required=False, default=5,
        help="Max/min value to plot the Wigner function for")
    parser.add_argument('--truncation', type=int, required=False, default=40,
        help="The truncation to use when calculating the states", metavar='T')
    parser.add_argument('--imsize', type=int, required=False, default=400,
        help="The size of the images to generate (in pixels)", metavar='SIZE')
    parser.add_argument('--dpi', type=int, required=False, default=192,
        help=("The DPI of your monitor (if not correct, images will be the "
        "wrong size)"))
    parser.add_argument('--verbose', required=False, default=False,
        action='store_true',
        help="Print progress information as the script runs")

    clas = parser.parse_args()
    VERBOSE = clas.verbose

    xvec = np.linspace(-clas.xlim, clas.xlim, 200)
    data, labels, params = gen_wigner(clas.training, clas.truncation, xvec)
    save_data("training", data, xvec, labels, params, clas.imsize, clas.dpi)
