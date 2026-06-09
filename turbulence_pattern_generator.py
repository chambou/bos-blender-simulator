import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from pathlib import Path

output_file = Path("data") / "turbulence_screen.tiff"

def field(size=200):
    """
    Generate a 2D random field with a power-law spectrum.

    The field is created in Fourier space using complex Gaussian noise
    and a spectral density proportional to k^-6 (since spectrum = k^-3
    and is applied to the amplitude).

    Parameters
    ----------
    size : int
        Number of pixels along each dimension.

    Returns
    -------
    np.ndarray
        Normalized 2D field with values in approximately [-1, 1].
    """

    # Spatial frequency coordinates
    kx = np.fft.fftfreq(size).reshape(-1, 1)
    ky = np.fft.fftfreq(size).reshape(1, -1)

    # Squared radial frequency
    k2 = kx**2 + ky**2

    # Avoid division by zero at the origin
    k2[0, 0] = 1

    # Power-law spectrum
    spectrum = k2 ** (-3.0)

    # Complex Gaussian white noise in Fourier space
    noise = np.random.randn(size, size) + 1j * np.random.randn(size, size)

    # Generate the field by inverse FFT
    f = np.fft.ifft2(noise * np.sqrt(spectrum)).real

    # Remove mean value
    f -= f.mean()

    # Normalize to [-1, 1]
    f /= np.max(np.abs(f))

    return f.astype(np.float32)


# --------------------------------------------------
# Generate a synthetic turbulence-like phase screen
# --------------------------------------------------
f = field()

# Remove residual offset
f -= np.mean(f)

# Normalize again for safety
f /= (np.max(np.abs(f)) + 1e-8)

# Scale amplitude to ensure visible contrast
f *= 0.5

# --------------------------------------------------
# Save as 32-bit floating-point TIFF
# --------------------------------------------------
output_file = "test.tiff"

tiff.imwrite(output_file, f.astype(np.float32))

print(f"Field saved to: {output_file}")