import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
######## geometry sampling #########

def rotate_vector(vector, angle):
    """Rotate a vector by a given angle"""
    return np.dot(vector, np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]))

def vec_vec_intersection(v1, p1, v2, p2):
    """Find intersection between two lines defined by vectors and points"""
    return p1 + (np.cross(p2-p1, v2/np.linalg.norm(v2)) / np.cross(v1/np.linalg.norm(v1), v2/np.linalg.norm(v2))) * v1

def quadrangle_self_intersection(vertices):
    """Check if a quadrangle has self-intersections"""
    # Check for self-intersections
    has_intersection = False
    
    # Check each pair of non-adjacent edges
    def segments_intersect(p1, p2, p3, p4):
        """Check if line segments (p1,p2) and (p3,p4) intersect"""
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
        return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)
    
    # Check intersection between edges (0,1) and (2,3)
    if segments_intersect(vertices[0], vertices[1], 
                        vertices[2], vertices[3]):
        has_intersection = True
        
    # Check intersection between edges (1,2) and (3,0)
    if segments_intersect(vertices[1], vertices[2], 
                        vertices[3], vertices[0]):
        has_intersection = True

    return has_intersection

# Check if any rounded points are outside the original quadrangle
def point_in_polygon(point, vertices):
    x, y = point
    n = len(vertices)
    inside = False
    j = n - 1
    for i in range(n):
        if ((vertices[i][1] > y) != (vertices[j][1] > y) and
            x < (vertices[j][0] - vertices[i][0]) * (y - vertices[i][1]) /
            (vertices[j][1] - vertices[i][1]) + vertices[i][0]):
            inside = not inside
        j = i
    return inside

def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def segments_intersect(p1, p2, p3, p4):
    return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)

def generate_polygon(num_vertices=4):
    """
    Generate a random polygon (triangle or quadrangle) without self-intersections in [0, 1] domain.
    Returns vertices as a numpy array of shape (num_vertices, 2).
    """
    while True:
        # Calculate vertices
        v1 = np.array([-0.5, -0.5])
        v2 = np.array([0.5, -0.5])
        
        vertices = [v1, v2]
        
        for _ in range(num_vertices - 2):
            x = np.random.uniform(-0.8, 0.8)
            y = np.random.uniform(-0.2 if num_vertices == 4 else 0.1, 0.8)
            vertices.append(np.array([x, y]))

        vertices = np.array(vertices)
        
        has_intersection = quadrangle_self_intersection(vertices) if num_vertices == 4 else False

        # Calculate area of the polygon
        area = 0
        for i in range(num_vertices):
            j = (i + 1) % num_vertices
            area += abs(np.cross(vertices[j] - vertices[0], vertices[i] - vertices[0])) / 2

        # Check if area is too small
        if (num_vertices == 4 and area < 0.35) or (num_vertices == 3 and area < 0.2):
            has_intersection = True
        
        if not has_intersection:
            # Calculate center point as mean of vertices
            center = np.mean(vertices, axis=0)
            
            # Calculate line segments through center
            lines = []
            vectors = []
            intermidiate_vectors = []
            rounded_quadrangles = []
            
            for v in vertices:
                vector = v - center
                normalized_vector = vector / np.linalg.norm(vector)
                vectors.append(normalized_vector)
                end_point = center + normalized_vector * 2
                lines.append([center, end_point])

            angle_offsets = np.array([np.random.uniform(-np.pi/6, np.pi/6) for _ in range(num_vertices)])
            angle_offset_sum = np.sum(np.abs(angle_offsets))
            desired_angle_offset_sum = np.random.uniform(0.0, 0.5)
            scale_factor = desired_angle_offset_sum / angle_offset_sum
            angle_offsets = angle_offsets.copy() * scale_factor

            result_angle_offsets = []

            for i_k in range(num_vertices):
                vector_1 = vectors[i_k]
                vector_2 = vectors[(i_k+1)%num_vertices]

                base_line_segment_v1 = vertices[i_k]
                base_line_segment_v2 = vertices[(i_k+1)%num_vertices]

                base_line_segment_vector = base_line_segment_v2 - base_line_segment_v1
                base_line_segment_normalized_vector = base_line_segment_vector / np.linalg.norm(base_line_segment_vector)

                base_line_center = (base_line_segment_v1 + base_line_segment_v2) / 2

                angle_offset = angle_offsets[i_k]
                rotated_vector_1 = rotate_vector(base_line_segment_normalized_vector, angle_offset)

                vector_to_center =  base_line_center - center
                vector_to_center_normalized = vector_to_center / np.linalg.norm(vector_to_center)

                line_offset = np.random.uniform(0.25, 0.5)

                offset_point = base_line_center + vector_to_center_normalized * line_offset

                # Generate 4 interpolated vectors between vector_1 and vector_2
                interpolated_vectors = []
                rounded_quadrangle_points = []

                ts = np.array([np.random.uniform(0.1, 0.3), np.random.uniform(0.00, 0.25), np.random.uniform(0.75, 1), np.random.uniform(0.6, 0.9)])
                for i, t in enumerate(ts):  # 4 points excluding endpoints
                    # Linear interpolation between vectors
                    vector_1_normalized = vector_1 / np.linalg.norm(vector_1)   
                    vector_2_normalized = vector_2 / np.linalg.norm(vector_2)
                    interp_vector = vector_1_normalized + t * (vector_2_normalized - vector_1_normalized)
                    # Normalize the interpolated vector
                    interp_vector = interp_vector / np.linalg.norm(interp_vector)
                    interpolated_vectors.append(interp_vector)
                    
                    if i in [0, 3]:
                        end_point = vec_vec_intersection(interp_vector, center, rotated_vector_1, offset_point)
                    else:
                        end_point = center + interp_vector * 2

                    rounded_quadrangle_points.append(end_point)

                    lines.append([center, end_point])

                intermidiate_vectors.append(interpolated_vectors)

                # Check if the rounded quadrangle has self-intersections
                rounded_quadrangle_self_intersection = quadrangle_self_intersection(rounded_quadrangle_points)

                if rounded_quadrangle_self_intersection == False:
                    rounded_quadrangle_lines = []
                    line_intersection = False
                    for i in range(4):
                        rounded_quadrangle_line = [rounded_quadrangle_points[i], rounded_quadrangle_points[(i+1)%4]]
                        for j in range(num_vertices):
                            main_quadrangle_line = [vertices[j], vertices[(j+1)%num_vertices]]
                            if segments_intersect(rounded_quadrangle_line[0], rounded_quadrangle_line[1], 
                                                main_quadrangle_line[0], main_quadrangle_line[1]):
                                line_intersection = True
                                break
                        rounded_quadrangle_lines.append(rounded_quadrangle_line)

                    if line_intersection == False:
                        rounded_quadrangles.append(rounded_quadrangle_lines)
                        result_angle_offsets.append(angle_offsets[i_k])   

                result_angle_offset_sum = np.sum(np.sin(np.abs(result_angle_offsets)))

            return vertices, center, lines, rounded_quadrangles, result_angle_offset_sum

######## SDF computation #########

# Using the signed_distance function from previous example
def point_to_line_distance(points, line_start, line_end):
    """Calculate signed distance from points to line segment"""
    line_vec = line_end - line_start
    point_vec = points - line_start
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    point_vec_scaled = point_vec / line_len
    t = np.dot(point_vec_scaled, line_unitvec)
    t = np.clip(t, 0.0, 1.0)
    nearest = line_start + t[:, np.newaxis] * line_vec
    dist = np.linalg.norm(points - nearest, axis=1)
    return dist

def point_in_triangle(points, v1, v2, v3):
    """Check if points are inside triangle using barycentric coordinates"""
    def sign(p1, p2, p3):
        return (p1[:, 0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[:, 1] - p3[1])
    
    d1 = sign(points, v1, v2)
    d2 = sign(points, v2, v3)
    d3 = sign(points, v3, v1)

    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)

    return ~(has_neg & has_pos)

def triangle_center(v1, v2, v3):
    """Calculate centroid of triangle"""
    return (v1 + v2 + v3) / 3.0

def point_in_quadrangle(points, vertices):
    """
    Check if points are inside quadrangle using winding number algorithm.
    
    Parameters:
    points: np.array([[x1, y1], [x2, y2], ...]) - the points to check
    vertices: np.array([[x1,y1], [x2,y2], [x3,y3], [x4,y4]]) - quadrangle vertices in CCW order
    
    Returns:
    np.array(bool): True for points inside, False otherwise
    """
    def is_left(p0, p1, p2):
        """
        Test if points p2 are left/on/right of line from p0 to p1.
        Returns:
        > 0 for p2 left of the line
        = 0 for p2 on the line
        < 0 for p2 right of the line
        """
        return ((p1[0] - p0[0]) * (p2[:, 1] - p0[1]) - 
                (p2[:, 0] - p0[0]) * (p1[1] - p0[1]))

    wn = np.zeros(len(points), dtype=int)  # winding number counter

    # Loop through all edges of the polygon
    for i in range(len(vertices)):
        # Get current and next vertex
        current = vertices[i]
        next = vertices[(i + 1) % len(vertices)]

        # Test if points are left/on/right of an edge
        mask = (current[1] <= points[:, 1]) & (next[1] > points[:, 1])  # an upward crossing
        wn[mask & (is_left(current, next, points) > 0)] += 1  # point left of edge

        mask = (current[1] > points[:, 1]) & (next[1] <= points[:, 1])  # a downward crossing
        wn[mask & (is_left(current, next, points) < 0)] -= 1  # point right of edge

    return wn != 0

def signed_distance_quadrangle(points, v1, v2, v3, v4, corner_radius=0.1):
    """Calculate signed distance from points to quadrangle"""
    d1 = point_to_line_distance(points, v1, v2)
    d2 = point_to_line_distance(points, v2, v3)
    d3 = point_to_line_distance(points, v3, v4)
    d4 = point_to_line_distance(points, v4, v1)
    
    min_dist = np.minimum.reduce([d1, d2, d3, d4])
    is_inside = point_in_quadrangle(points, [v1, v2, v3, v4])
    
    dist = np.where(is_inside, min_dist, -min_dist)

    return 0.5/(1 + np.exp(-20*(dist + corner_radius))) # 0.5 for not to be more then 1

def signed_distance_triangle(points, v1, v2, v3, corner_radius=0.1):
    """Calculate signed distance from points to triangle"""
    d1 = point_to_line_distance(points, v1, v2)
    d2 = point_to_line_distance(points, v2, v3)
    d3 = point_to_line_distance(points, v3, v1)

    min_dist = np.minimum.reduce([d1, d2, d3])
    is_inside = point_in_triangle(points, v1, v2, v3)

    dist = np.where(is_inside, min_dist, -min_dist)

    return 0.5/(1 + np.exp(-20*(dist + corner_radius))) # 0.5 for not to be more then 1


def signed_distance_sum(points, shapes):
    """Calculate signed distance sum from points to shapes"""
    sdf_sum = 0
    for shape in shapes:
        v1 = shape[0][0]
        v2 = shape[1][0]
        v3 = shape[2][0]
        if len(shape) == 3:
            sdf_sum += signed_distance_triangle(points, v1, v2, v3)
        else:
            v4 = shape[3][0]
            sdf_sum += signed_distance_quadrangle(points, v1, v2, v3, v4)
    return sdf_sum

def generate_and_calculate_signed_distance(num_vertices=4, noise_scale=0.03, noise_scale_z=0.04, points_per_side=64):
    start_time = time.time()
    
    # Generate random quadrangle vertices
    polygon = generate_polygon(num_vertices=num_vertices)
    vertices, center, lines, rounded_quadrangles, desired_angle_offset_sum = polygon

    # print(f"Time taken to generate polygon: {time.time() - start_time:.2f} seconds")
    # print(f"Desired angle offset sum: {desired_angle_offset_sum:.2f}")
    # Create list of vertex pairs for the polygon edges
    edges = []
    for i in range(len(vertices)):
        edges.append([vertices[i], vertices[(i+1) % len(vertices)]])
    rounded_quadrangles.append(edges)

    # Create a grid of points
    x = np.linspace(-1, 1, points_per_side)
    y = np.linspace(-1, 1, points_per_side)
    X, Y = np.meshgrid(x, y)
    
    # Add small random noise to coordinates
    X_noisy = X + np.random.normal(0, noise_scale, X.shape) 
    Y_noisy = Y + np.random.normal(0, noise_scale, Y.shape)

    # Calculate signed distance for each point
    points = np.stack([X_noisy.flatten(), Y_noisy.flatten()], axis=1)
    Z = signed_distance_sum(points, rounded_quadrangles).reshape(X.shape)

    # Add small random noise to Z values
    Z = Z + np.random.normal(-noise_scale_z/2, noise_scale_z/2, Z.shape)

    end_time = time.time()
    # print(f"Total time taken: {end_time - start_time:.2f} seconds")

    return X_noisy, Y_noisy, Z, rounded_quadrangles, desired_angle_offset_sum

######## Dataset generation #########

def save_dataset(dir, num_samples=1000, num_vertices=4, points_per_side=64):
    for sample_idx in tqdm(range(num_samples)):
        # Generate new random polygon and calculate everything
        # (assuming the polygon generation code is defined above)
        X, Y, Z, rounded_quadrangles, desired_angle_offset_sum = generate_and_calculate_signed_distance(
            num_vertices=num_vertices, points_per_side=points_per_side
        )
        # Store desired_angle_offset_sum
        if not hasattr(save_dataset, 'angle_offset_sums'):
            save_dataset.angle_offset_sums = []
        save_dataset.angle_offset_sums.append(desired_angle_offset_sum)
        
        # Prepare data for saving
        # Flatten X, Y into coordinate pairs and Z into values
        points = np.column_stack((X.flatten(), Y.flatten()))
        values = Z.flatten()
        
        # Save to CSV files
        sample_name = f'sample_{sample_idx:04d}'
        
        # Save points and their function values
        data = np.column_stack((points, values))
        np.savetxt(f'{dir}/{sample_name}_points.csv', data, 
                delimiter=',', 
                header='x,y,z', 
                comments='')
        
        # Save the scalar target value
        np.savetxt(f'{dir}/{sample_name}_target.csv', 
                np.array([desired_angle_offset_sum]), 
                delimiter=',',
                header='desired_angle_offset_sum',
                comments='')

    print('Dataset generation complete!')
    
    # Plot histogram of desired_angle_offset_sum values
    text_formula = r"$L_{orgz}$"
    plt.figure(figsize=(10, 6))
    plt.hist(save_dataset.angle_offset_sums, bins=30, edgecolor='black')
    plt.title(f'Histogram of {text_formula}')
    plt.xlabel(text_formula)
    plt.ylabel('Frequency')
    plt.show()

######## visualization #########

def plot_example_polygon(filename=None):
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()
    
    for i in range(4):
        # triangle = generate_triangle()
        polygon = generate_polygon(num_vertices=4)
        vertices, center, lines, rounded_quadrangles, desired_angle_offset_sum = polygon
        vertices_plot = np.vstack([vertices, vertices[0]])
        
        axes[i].plot(vertices_plot[:, 0], vertices_plot[:, 1], 'b-', linewidth=2)
        axes[i].fill(vertices[:, 0], vertices[:, 1], alpha=0.3)
        axes[i].scatter(vertices[:, 0], vertices[:, 1], c='red', s=100)

        axes[i].scatter(center[0], center[1], c='green', s=100, marker='*')
        
        # Add vertex labels
        for j, (x, y) in enumerate(vertices):
            axes[i].annotate(f'v{j}', (x, y), xytext=(5, 5), 
                           textcoords='offset points')
        
        # for line in lines:
        #     axes[i].plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], 'r-', linewidth=2)

        for rounded_quadrangle in rounded_quadrangles:
            for line in rounded_quadrangle:
                axes[i].plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], 'g-', linewidth=2)
        
        axes[i].grid(True)
        axes[i].set_xlim(-1.1, 1.1)
        axes[i].set_ylim(-1.1, 1.1)
        # axes[i].set_xlim(-3.1, 3.1)
        # axes[i].set_ylim(-3.1, 3.1)
        axes[i].set_aspect('equal')
        text_formula = r"$L_{orgz}$"
        axes[i].set_title(f'Quadrangle {i+1}, {text_formula}: {desired_angle_offset_sum:.2f}')
    
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def plot_sdf_surface(X, Y, Z, rounded_quadrangles = [], title = '', filename=None):

        # Call the function
    # X, Y, Z, rounded_quadrangles, desired_angle_offset_sum = generate_and_calculate_signed_distance()

    # Create plots
    fig = plt.figure(figsize=(20, 5))

    # 2D contour plot
    ax1 = fig.add_subplot(131)
    contour = ax1.contourf(X, Y, Z, levels=50, cmap='RdBu')
    for quadrangle in rounded_quadrangles:
        if len(quadrangle) == 3:
            v1, v2, v3 = quadrangle
            ax1.plot([v1[0][0], v2[0][0], v3[0][0], v1[0][0]], 
                [v1[0][1], v2[0][1], v3[0][1], v1[0][1]], 
                'k-', linewidth=2)
        else:
            v1, v2, v3, v4 = quadrangle
            ax1.plot([v1[0][0], v2[0][0], v3[0][0], v4[0][0], v1[0][0]], 
                [v1[0][1], v2[0][1], v3[0][1], v4[0][1], v1[0][1]], 
                'k-', linewidth=2)
    plt.colorbar(contour, ax=ax1)
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_title('2D Contour Plot of Signed Distance')
    ax1.set_aspect('equal')

    # 3D surface plot
    ax2 = fig.add_subplot(132, projection='3d')
    surf = ax2.plot_surface(X, Y, Z, cmap='RdBu')
    plt.colorbar(surf, ax=ax2)
    ax2.set_title('3D Surface Plot of Signed Distance')

    # 2D contour plot with more levels
    ax3 = fig.add_subplot(133)
    contour = ax3.contour(X, Y, Z, levels=50, cmap='RdBu')
    for quadrangle in rounded_quadrangles:
        if len(quadrangle) == 3:
            v1, v2, v3 = quadrangle
            ax3.plot([v1[0][0], v2[0][0], v3[0][0], v1[0][0]], 
                [v1[0][1], v2[0][1], v3[0][1], v1[0][1]], 
                'k-', linewidth=2)
        else:
            v1, v2, v3, v4 = quadrangle
            ax3.plot([v1[0][0], v2[0][0], v3[0][0], v4[0][0], v1[0][0]], 
                [v1[0][1], v2[0][1], v3[0][1], v4[0][1], v1[0][1]], 
                'k-', linewidth=2)
    plt.colorbar(contour, ax=ax3)
    ax3.set_title('2D Contour Lines')
    ax3.set_aspect('equal')
    ax3.set_xlim(-1, 1)
    ax3.set_ylim(-1, 1)
    fig.suptitle(f'Signed Distance Function Plots, {title}', fontsize=16)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()
