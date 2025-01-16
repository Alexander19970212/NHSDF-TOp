import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Ellipse, Polygon
from tqdm import tqdm
import gmsh

def triag_area(point1, point2, point3):
    mat = np.array([[point1[0], point1[1], 1],
                    [point2[0], point2[1], 1],
                    [point3[0], point3[1], 1]])
    
    return 0.5*np.abs(np.linalg.det(mat))

class LoadedMesh2D:
    """
    Class for loading mesh from filename.msh
    self.q   is an array with nodes coords
    self.me  is an array with indeces of coords for tetraidal elements
    self.volumes is an array with volumes for each element.
    """
    
    def __init__(self, filename):

        gmsh.initialize()
        gmsh.open(filename)
        
        entities = gmsh.model.getEntities()
        node_coords = []
        elem_node_tags = []
        elem_tags = []
        node_tags = []
        
        i = 0
        
        for e in entities:
            dim = e[0]
            tag = e[1]

            nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes(dim, tag)

            for i, nodetag in enumerate(nodeTags): # nodes
                node_tags.append(nodetag)
                node_coords.append(nodeCoords[i*3:i*3+2])


            if dim == 2: 

                elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim, tag)
            
                if elemTypes[0] == 2: # triagunal
                    elem_node_tags = np.array(elemNodeTags).reshape(int(np.array(elemNodeTags)[0].shape[0]/3), 3)
                    elem_tags = elemTags[0]
            
                i += 1
        
        gmsh.finalize()
        
        self.node_tags = np.array(node_tags) - 1 # indeces start from 1
        self.q = np.array(node_coords)
        
        self.elem_tags = np.array(elem_tags)
        self.me = np.array(elem_node_tags) - 1 # indeces start from 1

        # assemle volume list for each component
        area_list = []
        print("Compute areas ...")
        for element in tqdm(self.me):
            area = triag_area(self.q[int(element[0])], self.q[int(element[1])], self.q[int(element[2])])
            area_list.append(area)

        centroid_list = []
        for element in tqdm(self.me):
            centroid = self.q[element].sum(0)/3
            centroid_list.append(centroid)

        self.centroids = np.array(centroid_list)        
        self.areas = np.array(area_list)
        print('Whole area', self.areas.sum())

    def plot(self):
        fig = plt.figure()
        x = self.q[:, 0]
        y = self.q[:, 1]
        triangulation = mtri.Triangulation(x.ravel(), y.ravel(), self.me)
        # plt.scatter(x, y, color='red')
        plt.triplot(triangulation) #, 'g-h')
        
        plt.axis('equal')
        plt.show()

    def plot_topology(self, xPhys, geometry_features = None, filename=None):
        # fig = plt.figure(figsize=(12, 4))
        
        x = self.q[:, 0]
        y = self.q[:, 1]
        triangulation = mtri.Triangulation(x.ravel(), y.ravel(), self.me)
        fig = plt.figure(figsize=(6, 6 * (y.max() / x.max())))
        
        plt.xlim(0, x.max())
        plt.ylim(0, y.max())
        plt.tripcolor(triangulation, facecolors=xPhys, cmap='gray_r')
        plt.axis('equal')

        if geometry_features is not None:
            for geometry_feature in geometry_features:
                geometry_type = geometry_feature[0]
                if geometry_type == "ellipse":
                    a = geometry_feature[1]
                    b = geometry_feature[2]
                    center = geometry_feature[3]
                    rotation = geometry_feature[4]
                    rotation = rotation * 180 / np.pi
                    ellipse = Ellipse(center, 2*a, 2*b, angle=rotation, fill=False, color='red', linewidth=2)
                    plt.gca().add_patch(ellipse)

                elif geometry_type == "polygon":
                    vertices = geometry_feature[1]
                    radiuses = geometry_feature[2]
                    polygon = Polygon(vertices, fill=False, color='red', linewidth=4)
                    plt.gca().add_patch(polygon)

        if filename is not None:
            plt.gca().set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.axis('off')
            plt.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.show()

    def plot_displacement(self, u):
        d_x = u[::2]
        d_y = u[1::2]

        print("max_disp_x: ", d_x.max())
        print("max_disp_y: ", d_y.max())

        fig = plt.figure(figsize=(12, 4))
        x = self.q[:, 0]+d_x
        y = self.q[:, 1]+d_y
        triangulation = mtri.Triangulation(x.ravel(), y.ravel(), self.me)
        # plt.scatter(x, y, color='red')
        plt.triplot(triangulation) #, 'g-h')
        plt.axis('equal')
        plt.show()

class LoadedMesh2D_ext(LoadedMesh2D):
    def plot_topology(self, xPhys, vmax, vmin, ax):
        # fig = plt.figure(figsize=(12, 4))
        x = self.q[:, 0]
        y = self.q[:, 1]
        triangulation = mtri.Triangulation(x.ravel(), y.ravel(), self.me)
        # plt.scatter(x, y, color='red')
        # plt.triplot(triangulation, mask = x > 0.5) #, 'g-h')
        tcf = ax.tripcolor(triangulation, facecolors=xPhys, vmax = vmax, vmin = vmin)
        # plt.axis('equal')
        # plt.show()
        # tcf = ax.tricontourf(triangulation, xPhys, vmax = vmax, vmin = vmin)
        return tcf  
