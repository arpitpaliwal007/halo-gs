"""HaloGS: physically-grounded glow decomposition for 3D Gaussian Splatting.

Observed night image  I  =  clip( S + k_APSF(q, T) * E )

  S : glow-free scene render (standard SH Gaussians)
  E : rendered emission map (per-Gaussian emission attribute, same geometry)
  k : atmospheric point spread function (Narasimhan & Nayar 2003), 2 parameters

The scene is parallax-consistent across views; the glow is a convolution of the
*projected* emitters, so it moves with the camera rather than with surfaces.
That asymmetry is what lets multi-view training separate the two without any
glow-free supervision.
"""

__version__ = "0.1.0"
