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

    def plot_topology(self, xPhys, image_size=12,
                      zoom_center = None, zoom_radius = None, zoom_factor = 3,
                      ellipse_color = 'red',
                      geometry_features = None, filename=None):
        # fig = plt.figure(figsize=(12, 4))
        
        x = self.q[:, 0]
        y = self.q[:, 1]
        triangulation = mtri.Triangulation(x.ravel(), y.ravel(), self.me)
        fig, ax = plt.subplots(figsize=(image_size, image_size * (y.max() / x.max())))
        
        ax.set_xlim(0, x.max())
        ax.set_ylim(0, y.max())
        ax.tripcolor(triangulation, facecolors=xPhys, cmap='gray_r')
        ax.set_aspect('equal')

        if geometry_features is not None:
            for geometry_feature in geometry_features:
                geometry_type = geometry_feature[0]
                if geometry_type == "ellipse":
                    a = geometry_feature[1]
                    b = geometry_feature[2]
                    center = geometry_feature[3]
                    rotation = geometry_feature[4]
                    rotation = rotation * 180 / np.pi
                    ellipse = Ellipse(center, 2*a, 2*b, angle=rotation, fill=False, color=ellipse_color, linewidth=5)
                    ax.add_patch(ellipse)

                elif geometry_type == "polygon":
                    vertices = geometry_feature[1]
                    radiuses = geometry_feature[2]
                    line_segments = geometry_feature[3]
                    arc_segments = geometry_feature[4]

                    polygon = Polygon(vertices, fill=False, color='blue', linewidth=0.5)
                    ax.add_patch(polygon)

                    for start, end in line_segments:
                        ax.plot([start[0], end[0]], [start[1], end[1]], 'g-', linewidth=3)

                    # Plot arc segments
                    for center, start_angle, end_angle, radius in arc_segments:
                        # Calculate angles for arc
                        
                        # Ensure we draw the shorter arc
                        if abs(end_angle - start_angle) > np.pi:
                            if end_angle > start_angle:
                                start_angle += 2*np.pi
                            else:
                                end_angle += 2*np.pi
                                
                        # Create points along arc
                        theta = np.linspace(start_angle, end_angle, 100)
                        x = center[0] + radius * np.cos(theta)
                        y = center[1] + radius * np.sin(theta)
                        ax.plot(x, y, 'r-', linewidth=3)

        if zoom_center is not None:
            color = 'green'
            from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
            axins = zoomed_inset_axes(ax, zoom_factor, loc='upper right', borderpad=1)  # zoom-factor: 3, location: upper-right
            axins.spines['top'].set_color(color)
            axins.spines['top'].set_linewidth(2)
            axins.spines['bottom'].set_color(color)
            axins.spines['bottom'].set_linewidth(2)
            axins.spines['left'].set_color(color)
            axins.spines['left'].set_linewidth(2)
            axins.spines['right'].set_color(color)
            axins.spines['right'].set_linewidth(2)
            axins.set_xlim(zoom_center[0] - zoom_radius, zoom_center[0] + zoom_radius)
            axins.set_ylim(zoom_center[1] - zoom_radius, zoom_center[1] + zoom_radius)
            axins.tripcolor(triangulation, facecolors=xPhys, cmap='gray_r')
            axins.triplot(triangulation, color='black', linewidth=0.5)
            for i, triangle in enumerate(triangulation.triangles):
                centroid_x = np.mean(triangulation.x[triangle])
                centroid_y = np.mean(triangulation.y[triangle])
                offset_scale = 0.9
                scaled_zoom_radius = zoom_radius * offset_scale
                if (zoom_center[0] - scaled_zoom_radius <= centroid_x <= zoom_center[0] + scaled_zoom_radius) and \
                   (zoom_center[1] - scaled_zoom_radius <= centroid_y <= zoom_center[1] + scaled_zoom_radius):
                    rho_e = np.round(xPhys[i], 3)
                    e = i
                    rho_symbol = r'$\rho_{' + str(e) + '}$'
                    rho_value = r'$\bf{' + str(rho_e) + '}$'
                    text = f'{rho_symbol}\n{rho_value}'
                    axins.text(
                        centroid_x, 
                        centroid_y, 
                        text, 
                        color='orangered', 
                        fontsize=28, 
                        va='center', 
                        ha='center', 
                        bbox=dict(
                            facecolor='none', 
                            edgecolor='orangered', 
                            boxstyle='round,pad=0.1'
                        )
                    )

            if geometry_features is not None:
                for geometry_feature in geometry_features:
                    geometry_type = geometry_feature[0]
                    if geometry_type == "ellipse":
                        a = geometry_feature[1]
                        b = geometry_feature[2]
                        center = geometry_feature[3]
                        rotation = geometry_feature[4]
                        rotation = rotation * 180 / np.pi
                        ellipse = Ellipse(center, 2*a, 2*b, angle=rotation,
                                          fill=False, color=ellipse_color, linewidth=10)
                        axins.add_patch(ellipse)

            
            axins.set_aspect('equal')
            axins.set_xticks([])
            axins.set_yticks([])
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec=color, lw=2)

        if filename is not None:
            ax.set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            ax.axis('off')
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
