import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ---------------------------------------------------------------- load RGB
BACKGROUND = (255, 255, 255)  # colour used behind transparent areas (R, G, B)

raw = cv2.imread("car.png", cv2.IMREAD_UNCHANGED)  # keep alpha if present
if raw is None:
    raise FileNotFoundError("Could not read 'hello.png' - check the path.")

if raw.ndim == 2:                                   # greyscale file
    raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
if raw.dtype == np.uint16:                          # 16-bit PNG -> 8-bit
    raw = (raw / 257).astype(np.uint8)

if raw.shape[2] == 4:                               # RGBA: blend onto background
    rgb = cv2.cvtColor(raw[..., :3], cv2.COLOR_BGR2RGB).astype(np.float64)
    alpha = raw[..., 3:4].astype(np.float64) / 255.0
    bg = np.array(BACKGROUND, dtype=np.float64)
    img = rgb * alpha + bg * (1 - alpha)
else:
    img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB).astype(np.float64)  # H x W x 3


# ---------------------------------------------------------------- helpers
def convolve_rgb(image, kernel):
    """Convolve each of the R, G, B channels with the same 2D kernel."""
    big = max(kernel.shape) > 15  # FFT is much faster for large kernels
    out = np.empty_like(image)
    for c in range(3):
        if big:
            out[..., c] = signal.fftconvolve(image[..., c], kernel, mode="same")
        else:
            out[..., c] = signal.convolve2d(image[..., c], kernel,
                                            mode="same", boundary="symm")
    return out


def to_uint8(image, use_abs=False):
    if use_abs:
        image = np.abs(image)
    return np.clip(np.round(image), 0, 255).astype(np.uint8)


def shift_kernel(dy, dx, size=61):
    """Kernel that moves the image dy pixels down and dx pixels right.
    (Negative dy = up, negative dx = left.)  Note: convolution flips the
    kernel, so the '1' is placed at centre + (dy, dx)."""
    k = np.zeros((size, size))
    c = size // 2
    k[c + dy, c + dx] = 1
    return k


def gaussian_kernel(size, sigma):
    g = cv2.getGaussianKernel(size, sigma)
    return g @ g.T  # outer product -> 2D kernel that sums to 1


# ---------------------------------------------------------------- kernels
identity = np.zeros((3, 3)); identity[1, 1] = 1

S = 25  # shift distance in pixels
shifts = {
    "Shift left":            (0, -S),
    "Shift right":           (0,  S),
    "Shift up":              (-S, 0),
    "Shift down":            (S,  0),
    "Shift up-left":         (-S, -S),
    "Shift up-right":        (-S,  S),
    "Shift down-left":       (S,  -S),
    "Shift down-right":      (S,   S),
}

gauss_5 = gaussian_kernel(5, 1.0)
gauss_15 = gaussian_kernel(15, 3.0)
box_blur = np.ones((5, 5)) / 25.0

# High-pass = image - blurred image  (keeps only fine detail / edges)
highpass = identity.copy()
highpass = np.zeros((5, 5)); highpass[2, 2] = 1
highpass = highpass - gauss_5

# Sharpening = original + high-pass  (= 2*identity - gaussian)
sharpen = np.zeros((5, 5)); sharpen[2, 2] = 1
sharpen = sharpen + highpass

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=float)
sobel_y = sobel_x.T

prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]], dtype=float)
prewitt_y = prewitt_x.T

laplacian_4 = np.array([[0,  1, 0],
                        [1, -4, 1],
                        [0,  1, 0]], dtype=float)
laplacian_8 = np.array([[1,  1, 1],
                        [1, -8, 1],
                        [1,  1, 1]], dtype=float)
emboss = np.array([[-2, -1, 0],
                   [-1,  1, 1],
                   [ 0,  1, 2]], dtype=float)

# ---------------------------------------------------------------- apply all
results = [("Original", to_uint8(img)),
           ("Identity", to_uint8(convolve_rgb(img, identity)))]

for name, (dy, dx) in shifts.items():
    results.append((name, to_uint8(convolve_rgb(img, shift_kernel(dy, dx)))))

results += [
    ("Gaussian blur 5x5 (σ=1)",   to_uint8(convolve_rgb(img, gauss_5))),
    ("Gaussian blur 15x15 (σ=3)", to_uint8(convolve_rgb(img, gauss_15))),
    ("Box blur 5x5",              to_uint8(convolve_rgb(img, box_blur))),
    ("High-pass (detail only)",   to_uint8(convolve_rgb(img, highpass) + 128)),  # +128 so negatives are visible
    ("Sharpened (original + high-pass)", to_uint8(convolve_rgb(img, sharpen))),
]

# Edge detection: gradient in x and y, then magnitude
for name, kx, ky in [("Sobel", sobel_x, sobel_y), ("Prewitt", prewitt_x, prewitt_y)]:
    gx = convolve_rgb(img, kx)
    gy = convolve_rgb(img, ky)
    mag = np.hypot(gx, gy)
    results += [
        (f"{name} X (vertical edges)",   to_uint8(gx, use_abs=True)),
        (f"{name} Y (horizontal edges)", to_uint8(gy, use_abs=True)),
        (f"{name} magnitude",            to_uint8(mag)),
    ]

results += [
    ("Laplacian (4-neighbour)", to_uint8(convolve_rgb(img, laplacian_4), use_abs=True)),
    ("Laplacian (8-neighbour)", to_uint8(convolve_rgb(img, laplacian_8), use_abs=True)),
    ("Emboss",                  to_uint8(convolve_rgb(img, emboss) + 128)),
]

# ---------------------------------------------------------------- display
import textwrap

plt.style.use("default")           # ignore any dark/custom theme
FIG_BG = "#202020"                 # dark grey window background
TITLE_COLOR = "white"              # title text colour (always contrasts with FIG_BG)

cols = 6
rows = math.ceil(len(results) / cols)
# Keep the figure small enough to fit on a normal screen (18 x 10 inches max),
# otherwise the window is taller than the screen and the top row gets cut off.
fig, axes = plt.subplots(rows, cols, figsize=(18, 10),
                         facecolor=FIG_BG, layout="constrained")
fig.get_layout_engine().set(w_pad=0.05, h_pad=0.08)  # padding around the plots
axes = axes.ravel()

for ax, (title, im) in zip(axes, results):
    ax.imshow(im)          # RGB uint8 -> shown with correct colours
    ax.set_title(textwrap.fill(title, 24), fontsize=10,
                 fontweight="bold", color=TITLE_COLOR, pad=6)
    ax.axis("off")
for ax in axes[len(results):]:
    ax.axis("off")

plt.show()