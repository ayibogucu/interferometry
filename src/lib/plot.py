import sys

import numpy as np
from vispy import app, scene
from vispy.color import get_colormap
from vispy.geometry import MeshData


def plot_mesh(image: np.ndarray):
    h, w = image.shape
    z = image  # Use the image values as z-values

    # 1. Create a grid of vertices preserving the image scale.
    # x and y are pixel indices; z is taken directly from the image.
    x = np.arange(w)
    y = np.arange(h)
    xv, yv = np.meshgrid(x, y)
    vertices = np.column_stack((xv.ravel(), yv.ravel(), z.ravel()))

    # 2. Create faces for the mesh (two triangles per grid cell).
    faces = []
    for i in range(h - 1):
        for j in range(w - 1):
            idx0 = i * w + j
            idx1 = i * w + (j + 1)
            idx2 = (i + 1) * w + j
            idx3 = (i + 1) * w + (j + 1)
            # First triangle.
            faces.append([idx0, idx1, idx2])
            # Second triangle.
            faces.append([idx2, idx1, idx3])
    faces = np.array(faces)

    # 3. Create mesh data.
    mesh_data = MeshData(vertices=vertices, faces=faces)

    # 4. Compute vertex colors from z-values using the 'fire' colormap.
    cmap = get_colormap("jet")
    z_min, z_max = z.min(), z.max()
    # Normalize the z-values to the [0, 1] range.
    norm = (vertices[:, 2] - z_min) / (z_max - z_min)
    # Reshape to (N,1) so the colormap can broadcast properly.
    norm = norm[:, None]
    # Map normalized values to RGBA colors and cast to float32.
    vertex_colors = cmap.map(norm).astype(np.float32)

    # 5. Set the computed vertex colors into the mesh data.
    mesh_data.set_vertex_colors(vertex_colors)

    # 6. Set up the canvas and view.
    canvas = scene.SceneCanvas(keys="interactive", bgcolor="w", size=(800, 600))
    view = canvas.central_widget.add_view()
    view.camera = scene.TurntableCamera(up="z", fov=60)

    # 7. Create the mesh visual. (No need to pass vertex_colors here since they are in mesh_data.)
    mesh = scene.visuals.Mesh(meshdata=mesh_data, shading="smooth")
    view.add(mesh)

    canvas.show()
    if sys.flags.interactive == 0:
        app.run()


def plot_surface(image: np.ndarray):
    z = image

    canvas = scene.SceneCanvas(keys="interactive", bgcolor="w")
    view = canvas.central_widget.add_view()
    view.camera = scene.TurntableCamera(up="z", fov=60)

    # Simple surface plot example
    # x, y values are not specified, so assumed to be 0:50
    p1 = scene.visuals.SurfacePlot(z=z, color=(0.3, 0.3, 1, 1))
    view.add(p1)

    # p1._update_image()  # cheating.
    # cf = scene.filters.ZColormapFilter('fire', zrange=(z.max(), z.min()))
    # p1.attach(cf)

    # Add a 3D axis to keep us oriented
    # axis = scene.visuals.XYZAxis(parent=view.scene)

    canvas.show()
    if sys.flags.interactive == 0:
        app.run()
