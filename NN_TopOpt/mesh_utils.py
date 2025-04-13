import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Ellipse, Polygon
from tqdm import tqdm
import gmsh

import matplotlib.colors as mcolors

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

    def plot_topology(self, xPhys, von_mises=None, image_size=12,
                      zoom_center = None, zoom_radius = None, zoom_factor = 3,
                      ellipse_color = 'red',
                      geometry_features = None, filename=None):
        # fig = plt.figure(figsize=(12, 4))

        # if von_mises is not None:
        #     mask = xPhys > 0.1

        
        x = self.q[:, 0]
        y = self.q[:, 1]


        triangulation = mtri.Triangulation(x.ravel(), y.ravel(), self.me)
        fig, ax = plt.subplots(figsize=(image_size, image_size * (y.max() / x.max())))
        
        ax.set_xlim(0, x.max())
        ax.set_ylim(0, y.max())

        if von_mises is not None:
            # von_mises_masked = np.full(von_mises.shape, np.nan, dtype=float)
            # von_mises_masked[mask] = von_mises[mask]
            # cmap = plt.get_cmap('viridis')
            # colors = cmap(von_mises)
            # print(colors)
            # colors[:, 3] = xPhys  # Set alpha channel using xPhys: 0 is fully transparent, 1 is fully opaque
            # print(colors)
            xPhys = np.clip(xPhys, 0.0, 1.0)

            num_colors = 10
            markers = np.array([0, 500])
            markers = np.append(markers, np.linspace(2500, 7000, num_colors - 3))
            markers = np.append(markers, 8000)
            markers = np.append(markers, von_mises.max())
            discrete_cmap = plt.get_cmap('turbo', num_colors)
            norm = mcolors.BoundaryNorm(markers, num_colors)

            # tpc = ax.tripcolor(triangulation, facecolors=von_mises, alpha=xPhys,
            #                    cmap=discrete_cmap, norm=norm, shading='flat')
            

            # tpc = ax.tripcolor(triangulation, facecolors=von_mises, alpha=xPhys,
            #                    cmap="turbo", shading='flat')
            # Convert triangle-based von_mises values to node-based values by averaging
            node_vals = np.zeros(x.shape)
            counts = np.zeros(x.shape)
            # Accumulate triangle values to their respective nodes
            np.add.at(node_vals, self.me.ravel(), np.repeat(von_mises, self.me.shape[1]))
            np.add.at(counts, self.me.ravel(), 1)
            # Compute the average for each node
            node_vals = node_vals / counts

            #mask the nodes with xPhys < 0.3
            mask = xPhys < 0.3
            triangulation.set_mask(mask)

            von_mises_max = von_mises[~mask].max()
            von_mises_min = von_mises[~mask].min()

            print(von_mises_max, von_mises_min)

            # Create levels for the contour plot
            levels_percentages = np.array([0, 0.05, 0.08, 0.1, 0.12, 0.14, 0.16,  0.2, 0.4, 0.7, 0.8, 1])
            levels = levels_percentages * (von_mises_max - von_mises_min) + von_mises_min
            
            # Compute contour levels using quantile statistics on the node_vals distribution.
            # This approach divides the data into bins with equal probability mass, so regions with more frequent values
            # receive finer resolution in the contour plot.
            # n_levels = 12  # Adjust the number of contour levels as needed for finer resolution in dense areas
            # levels = np.percentile(node_vals, np.linspace(0, 100, n_levels))
            # # Remove any duplicate levels that may occur in low variance scenarios to ensure proper contouring.
            # levels = np.unique(levels)

            tpc = ax.tricontourf(triangulation, node_vals,
                                levels=levels,
                                cmap="turbo")

            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.2)
            cbar = fig.colorbar(tpc, cax=cax)
            cbar.set_label("Von Mises stress (Pa)", size=15)

            # ax.set_title("Von Mises stress", size=20)

            # Create an additional axes for the histogram inset
            # ax_hist = fig.add_axes([0.7, 0.2, 0.2, 0.2])  # [left, bottom, width, height] in figure fraction coordinates
            # Plot the histogram of von_mises values, flatten in case it's multidimensional
            # ax_hist.hist(von_mises.flatten(), bins=30, color='blue', edgecolor='black')
            # ax_hist.set_title("Distribution of Von Mises")
            # ax_hist.set_xlabel("Von Mises")
            # ax_hist.set_ylabel("Frequency")
            # tpc.set_array(np.array([]))
        else:
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
