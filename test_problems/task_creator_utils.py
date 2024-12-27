import gmsh
import os
import sys

sys.path.append(os.path.abspath('../NN_TopOpt'))

from mesh_utils import LoadedMesh2D 
import json

def create_task(poligone_vertices, task_name, constraints, loads):
    gmsh.initialize()
    model = gmsh.model()

    points = []
    lines = []

    for vertex in poligone_vertices:
        points.append(model.occ.add_point(vertex[0], vertex[1], 0))

    for i in range(len(points)):
        lines.append(model.occ.add_line(points[i], points[(i + 1) % len(points)]))

    face_curve = model.occ.add_curve_loop(lines)
    face = model.occ.add_plane_surface([face_curve])

    model.occ.synchronize()
    model.add_physical_group(dim=2, tags=[face])

    gmsh.option.setNumber('Mesh.MeshSizeMax', 0.005)

    #mesh generation
    model.mesh.generate(dim=2)

    # Creates  graphical user interface
    # if 'close' not in sys.argv:
        # gmsh.fltk.run()\
    gmsh.write(f"{task_name}.msh")
    
    # It finalize the Gmsh API
    gmsh.finalize()

    problem_list = {f"{task_name}": {"meshfile": f"test_problems/{task_name}.msh",
                              "fixed_x": constraints["fixed_x"],
                              "fixed_y": constraints["fixed_y"],
                              "fixed_xy": constraints["fixed_xy"],
                              "loads": loads}}

    try:
        with open('test_problems/problems_new.json', 'r') as fp:
            existing_problems = json.load(fp)
    except FileNotFoundError:
        existing_problems = {}

    existing_problems.update(problem_list)

    with open('problems.json', 'w') as fp:
        json.dump(existing_problems, fp)

    Th = LoadedMesh2D(f"{task_name}.msh")
    Th.plot()
