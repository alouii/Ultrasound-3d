import pyvista as pv

print("PyVista version:", pv.__version__)
grid = pv.UniformGrid(dimensions=(10, 10, 10))
print("Created grid:", grid)
plotter = pv.Plotter()
plotter.add_mesh(grid.outline(), color="red")
plotter.show()