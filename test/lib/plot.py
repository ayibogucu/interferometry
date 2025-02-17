import sys
from vispy import app, scene
import numpy as np


def plot_surface(image: np.ndarray):
    # image = np.load("./results/100.npy") * 20
    # # image = loadmat("./results/unwrapped_phase.mat")["s"] * -1
    h, w = image.shape
    z = image

    canvas = scene.SceneCanvas(keys="interactive", bgcolor="w")
    view = canvas.central_widget.add_view()
    view.camera = scene.TurntableCamera(up="z", fov=60)

    # Simple surface plot example
    # x, y values are not specified, so assumed to be 0:50
    p1 = scene.visuals.SurfacePlot(z=z, color=(0.3, 0.3, 1, 1))
    p1.transform = scene.transforms.MatrixTransform()
    p1.transform.scale([1 / (w - 1), 1 / (h - 1), 1 / (z.max() - z.min() - 1)])
    p1.transform.translate([-0.5, -0.5, 0])
    view.add(p1)

    # p1._update_image()  # cheating.
    # cf = scene.filters.ZColormapFilter('fire', zrange=(z.max(), z.min()))
    # p1.attach(cf)

    # Add a 3D axis to keep us oriented
    # axis = scene.visuals.XYZAxis(parent=view.scene)

    canvas.show()
    if sys.flags.interactive == 0:
        app.run()
