import numpy as np

# Using the signed_distance function from previous example
def point_to_line_distance(point, line_start, line_end):
    """Calculate signed distance from point to line segment"""
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    point_vec_scaled = point_vec / line_len
    t = np.dot(line_unitvec, point_vec_scaled)
    t = np.clip(t, 0.0, 1.0)
    nearest = line_start + t * line_vec
    dist = np.linalg.norm(point - nearest)
    return dist


def normalize(v): # + |
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def bisector_ray(seg1_start, seg1_end, seg2_start, seg2_end): # + -> |
    """
    Calculate the bisector ray of two line segments.
    
    Parameters:
    - seg1_start, seg1_end: numpy arrays representing the start and end of the first segment
    - seg2_start, seg2_end: numpy arrays representing the start and end of the second segment
    
    Returns:
    - bisector: numpy array representing the bisector ray
    """
    # Calculate direction vectors
    dir1 = normalize(seg1_end - seg1_start)
    dir2 = normalize(seg2_end - seg2_start)
    
    # Calculate bisector
    bisector = normalize(dir1 + dir2)
    return bisector

def orthogonal_vector(line_start, line_end): # + |
    """
    Calculate an orthogonal vector to a line segment.
    
    Parameters:
    - line_start, line_end: numpy arrays representing the start and end of the line segment
    
    Returns:
    - orthogonal: numpy array representing the orthogonal vector
    """
    # Calculate direction vector
    direction = line_end - line_start
    
    # Calculate orthogonal vector by rotating 90 degrees
    orthogonal = np.array([-direction[1], direction[0]])
    return orthogonal

def find_intersection(ray1_start, ray1_dir, ray2_start, ray2_dir): # + -> |
    """
    Find the intersection point of two rays in 2D.
    
    Parameters:
    - ray1_start: numpy array representing the start point of the first ray
    - ray1_dir: numpy array representing the direction vector of the first ray
    - ray2_start: numpy array representing the start point of the second ray
    - ray2_dir: numpy array representing the direction vector of the second ray
    
    Returns:
    - intersection: numpy array representing the intersection point, or None if no intersection
    """
    # Normalize direction vectors
    ray1_dir = normalize(ray1_dir)
    ray2_dir = normalize(ray2_dir)
    
    # Calculate the determinant
    det = ray1_dir[0] * ray2_dir[1] - ray1_dir[1] * ray2_dir[0]
    
    if np.isclose(det, 0):
        # Rays are parallel or collinear
        return None
    
    # Calculate the intersection point using Cramer's rule
    diff = ray2_start - ray1_start
    t1 = (diff[0] * ray2_dir[1] - diff[1] * ray2_dir[0]) / det
    intersection = ray1_start + t1 * ray1_dir
    
    return intersection

def maximal_rounding_radius(seg1_start, seg1_end, seg2_start, seg2_end): # + -> |
    """Calculate the maximal rounding radius for two non-parallel line segments."""
    bisector = bisector_ray(seg1_start, seg1_end, seg2_start, seg2_end)
    bisector_origin = find_intersection(seg1_start, seg1_end - seg1_start, seg2_start, seg2_end - seg2_start)
    orthogonal1 = orthogonal_vector(seg1_start, seg1_end)
    orthogonal2 = orthogonal_vector(seg2_start, seg2_end)
    
    intersection_cases = [[seg1_start, orthogonal1],
                            [seg1_end, orthogonal1], 
                            [seg2_start, orthogonal2], 
                            [seg2_end, orthogonal2]]
    
    distances = []
    point_intersections = []
    for case in intersection_cases:
        line_org, line_dir = case[0], case[1]
        point_intersection = find_intersection(bisector_origin, bisector, line_org, line_dir)

        point_distance = np.linalg.norm(point_intersection - line_org)
        point_intersections.append(point_intersection)
        distances.append(point_distance)

    # Find the index of the third minimal value in distances
    sorted_indices = np.argsort(distances)
    third_minimal_index = sorted_indices[2]

    max_radius = distances[third_minimal_index]
    center = point_intersections[third_minimal_index]
    
    return max_radius, center

def compute_perimeter(line_segments, arc_segments):
    """
    Compute the perimeter of a polygon consisting of line and arc segments.
    
    Parameters:
    - line_segments: list of tuples representing the line segments [(start_point, end_point), ...]
    - arc_segments: list of tuples representing the arc segments [(center, start_angle, end_angle, radius), ...]
    
    Returns:
    - perimeter: float, the total perimeter of the polygon
    """
    perimeter = 0.0
    line_perimeter = 0.0
    arc_perimeter = 0.0
    
    # Calculate the perimeter contribution from line segments
    for line_segment in line_segments:
        start_point, end_point = line_segment
        perimeter += np.linalg.norm(end_point - start_point)
        line_perimeter += np.linalg.norm(end_point - start_point)
    
    # Calculate the perimeter contribution from arc segments
    for arc_segment in arc_segments:
        center, start_angle, end_angle, radius = arc_segment
        angle_diff = abs(end_angle - start_angle)
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff
        arc_length = angle_diff * radius
        perimeter += arc_length
        arc_perimeter += arc_length

    return perimeter, line_perimeter, arc_perimeter

import numpy as np
import matplotlib.pyplot as plt


def get_corner_points(p1, p, p2, r): # + |
    """Get points where arcs meet lines for a corner."""
    v1 = normalize(p1 - p)
    v2 = normalize(p2 - p)
    
    angle = np.arccos(np.dot(v1, v2))
    d = r / np.tan(angle / 2)
    
    p1_new = p + v1 * d
    p2_new = p + v2 * d
    
    bisector = normalize(v1 + v2)
    center_dist = r / np.sin(angle / 2)
    center = p + bisector * center_dist
    
    start_angle = np.arctan2(p1_new[1] - center[1], p1_new[0] - center[0])
    end_angle = np.arctan2(p2_new[1] - center[1], p2_new[0] - center[0])
    
    if abs(end_angle - start_angle) > np.pi:
        if end_angle > start_angle:
            end_angle -= 2 * np.pi
        else:
            start_angle -= 2 * np.pi
            
    return p1_new, p2_new, center, start_angle, end_angle

def get_rounded_polygon_segments(vertices, radius): # + -> |
    """Get arc segments and line segments for a polygon with rounded corners."""
    num_vertices = len(vertices)
    line_segments = []
    arc_ends = []
    arc_segments = []
    v_p_distances = []
    
    for i in range(num_vertices):
        v1 = vertices[i - 1]
        v2 = vertices[i]
        v3 = vertices[(i + 1) % num_vertices]
        
        p1_new, p2_new, center, start_angle, end_angle = get_corner_points(v1, v2, v3, radius)
        
        arc_ends.append((p2_new, p1_new))
        arc_segments.append((center, start_angle, end_angle, radius))
        v_p_distances.append(np.linalg.norm(v2 - p2_new))

    arcs_intersection = False
    for i in range(num_vertices):
        v_1 = vertices[i - 1]
        v_2 = vertices[i]
        v_1_p_distance = v_p_distances[i - 1]
        v_2_p_distance = v_p_distances[i]

        segment_length = np.linalg.norm(v_2 - v_1)
        if v_1_p_distance + v_2_p_distance > segment_length:
            arcs_intersection = True
            break


    for i in range(num_vertices):
        line_segments.append((arc_ends[i][0], arc_ends[(i + 1) % num_vertices][1]))

    
    return line_segments, arc_segments, arcs_intersection

def get_rounded_polygon_segments_with_given_radiuses(vertices, radiuses): # + -> |
    """Get arc segments and line segments for a polygon with rounded corners."""
    num_vertices = len(vertices)
    line_segments = []
    arc_ends = []
    arc_segments = []
    v_p_distances = []
    
    for i in range(num_vertices):
        v1 = vertices[i - 1]
        v2 = vertices[i]
        v3 = vertices[(i + 1) % num_vertices]
        
        p1_new, p2_new, center, start_angle, end_angle = get_corner_points(v1, v2, v3, radiuses[(i - 1) % num_vertices])
        
        arc_ends.append((p2_new, p1_new))
        arc_segments.append((center, start_angle, end_angle, radiuses[(i - 1) % num_vertices]))
        v_p_distances.append(np.linalg.norm(v2 - p2_new))

    arcs_intersection = False
    for i in range(num_vertices):
        v_1 = vertices[i - 1]
        v_2 = vertices[i]
        v_1_p_distance = v_p_distances[i - 1]
        v_2_p_distance = v_p_distances[i]

        segment_length = np.linalg.norm(v_2 - v_1)
        if v_1_p_distance + v_2_p_distance > segment_length:
            arcs_intersection = True
            break


    for i in range(num_vertices):
        line_segments.append((arc_ends[i][0], arc_ends[(i + 1) % num_vertices][1]))

    
    return line_segments, arc_segments, arcs_intersection


def get_rounded_polygon_segments_rand_radius(vertices, min_radius = 0.01, max_radius_limit = 3): # + ->
    """Get arc segments and line segments for a polygon with rounded corners."""
    
    line_segments_cut, _,  arcs_intersection = get_rounded_polygon_segments(vertices, min_radius)
    
    if arcs_intersection:
        return [], [], True
    
    else:
        num_vertices = len(vertices)
        line_segments = []
        arc_segments = []
        arc_ends = []
        v_p_distances = []

        first_middle_segments = []
        second_middle_segments = [] 

        random_variables = np.random.uniform(0.2, 0.8, num_vertices)

        for i in range(num_vertices):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % num_vertices]
            v3 = vertices[(i + 2) % num_vertices]

            line_segment_1 = line_segments_cut[i]
            line_segment_2 = line_segments_cut[(i + 1) % num_vertices]
            line_segment_2 = line_segment_2[::-1]

            middle_point_1 = line_segment_1[0] + (line_segment_1[1] - line_segment_1[0])*random_variables[i]
            middle_point_2 = line_segment_2[0] + (line_segment_2[1] - line_segment_2[0])*(1 - random_variables[(i+1) % num_vertices])

            midd_seg_1 = [line_segment_1[1], middle_point_1]
            midd_seg_2 = [line_segment_2[1], middle_point_2]
      
            if i == 1:
                first_middle_segments.append(midd_seg_1)
                first_middle_segments.append(midd_seg_2)
      
            A1 = midd_seg_1[0]
            A2 = midd_seg_1[1]
            B1 = midd_seg_2[0]
            B2 = midd_seg_2[1]

            max_radius, center = maximal_rounding_radius(A1, A2, B1, B2)
            max_radius = min(max_radius, max_radius_limit)
            if np.random.uniform(0, 1) < 0.5:
                valid_radius = np.random.uniform(min_radius, max_radius)
            else:
                valid_radius = np.random.uniform(min_radius, min_radius + (max_radius - min_radius)*0.2)

            p1_new, p2_new, center, start_angle, end_angle = get_corner_points(v1, v2, v3, valid_radius)
        
            v_p_distances.append(np.linalg.norm(v2 - p2_new))
            arc_ends.append((p2_new, p1_new))
            arc_segments.append((center, start_angle, end_angle, valid_radius))

        # arcs_intersection = False
        for i in range(num_vertices):
            v_1 = vertices[i]
            v_2 = vertices[(i + 1) % num_vertices]
            v_1_p_distance = v_p_distances[i - 1]
            v_2_p_distance = v_p_distances[i]

            segment_length = np.linalg.norm(v_2 - v_1)
            if v_1_p_distance + v_2_p_distance > segment_length:
                # arcs_intersection = True
                print(f'arcs_intersection {segment_length - v_1_p_distance - v_2_p_distance}')
                

        for i in range(num_vertices):
            line_segments.append((arc_ends[i][0], arc_ends[(i + 1) % num_vertices][1]))
            
        return line_segments, arc_segments, False

def points_to_arc_distances(points, arc_center, start_angle, end_angle, radius): # + |
    """
    Calculate the minimal distances from a set of points to an arc segment.
    
    Parameters:
    - points: numpy array of shape (N, 2) representing the points [[x1, y1], [x2, y2], ...]
    - arc_center: numpy array representing the center of the arc [x, y]
    - start_angle: float, start angle of the arc in radians
    - end_angle: float, end angle of the arc in radians
    - radius: float, radius of the arc
    
    Returns:
    - min_distances: numpy array of shape (N,) containing the minimal distances from each point to the arc
    """
    # Calculate vectors from the center to each point
    # Check if the distance between angles is more than pi, then rechange it
    # print(start_angle, end_angle, end_angle - start_angle)
    if end_angle - start_angle < 0:
        start_angle, end_angle = end_angle, start_angle
        # print("rechange")
    centers_to_points = points - arc_center
    distances_to_center = np.linalg.norm(centers_to_points, axis=1)
    
    # Calculate angles of the points relative to the arc center
    points_angles = np.arctan2(centers_to_points[:, 1], centers_to_points[:, 0])
    
    # Normalize angles to be within the range [0, 2*pi)
    points_angles = points_angles % (2 * np.pi)
    start_angle = start_angle % (2 * np.pi)
    end_angle = end_angle % (2 * np.pi)
    
    # Check if each point's angle is within the arc's angular span
    if start_angle <= end_angle:
        within_arc = (start_angle <= points_angles) & (points_angles <= end_angle)
    else:
        within_arc = (points_angles >= start_angle) | (points_angles <= end_angle)
    
    # Calculate distances to the arc
    min_distances = np.empty(points.shape[0])
    min_distances[within_arc] = np.abs(distances_to_center[within_arc] - radius)
    
    # Calculate distances to the arc's endpoints for points outside the angular span
    start_point = arc_center + radius * np.array([np.cos(start_angle), np.sin(start_angle)])
    end_point = arc_center + radius * np.array([np.cos(end_angle), np.sin(end_angle)])
    distances_to_start = np.linalg.norm(points - start_point, axis=1)
    distances_to_end = np.linalg.norm(points - end_point, axis=1)
    
    min_distances[~within_arc] = np.minimum(distances_to_start[~within_arc], distances_to_end[~within_arc])
    
    return min_distances

import numpy as np

def are_points_in_polygon(points, vertices): # + |
    """
    Determine if a set of points are inside a polygon using the Ray Casting algorithm.
    
    Parameters:
    - points: numpy array of shape (M, 2) representing the points [[x1, y1], [x2, y2], ...]
    - vertices: numpy array of shape (N, 2) representing the vertices of the polygon
    
    Returns:
    - inside: numpy array of shape (M,) with boolean values, True if the point is inside the polygon, False otherwise
    """
    n = len(vertices)
    inside = np.zeros(points.shape[0], dtype=bool)

    for j, (x, y) in enumerate(points):
        p1x, p1y = vertices[0]
        for i in range(n + 1):
            p2x, p2y = vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside[j] = not inside[j]
            p1x, p1y = p2x, p2y

    return inside

def if_points_in_polygon(points, line_segments, arc_segments, base_vertices): # + -> |
    all_vertices = np.array(line_segments).reshape(-1, 2)
    inside = are_points_in_polygon(points, all_vertices)

    arc_centers = np.array([center for center, _, _, _ in arc_segments])
    # if_centers_inside = are_points_in_polygon(arc_centers, vertices)

    intermideate_line_segments = []
    for i in range(len(line_segments)):
        intermideate_line_segments.append([line_segments[i][1], line_segments[(i + 1) % len(line_segments)][0]])
    intermideate_line_segments = np.array(intermideate_line_segments)
    middle_points_intermideate_line_segments = intermideate_line_segments.mean(axis=1)

    arc_center_distances = []
    for i, (center, _, _, _) in enumerate(arc_segments):
        distance = np.linalg.norm(center - middle_points_intermideate_line_segments[i-1])
        arc_center_distances.append(distance)

    if_centers_inside = are_points_in_polygon(middle_points_intermideate_line_segments, base_vertices)
    # print(if_centers_inside)
    # inside = np.zeros(points.shape[0], dtype=bool)

    for i, (center, start_angle, end_angle, radius) in enumerate(arc_segments):
        # Calculate distances from points to arc center
        # distances = np.sqrt(np.sum((points - center) ** 2, axis=1))
        # Points inside circle are those with distance less than radius
        # points_in_circle = distances <= radius

        # Vector from center to point
        centers_to_points = points - center
        distances_to_center = np.linalg.norm(centers_to_points, axis=1)
        
        # Angles of points relative to center
        points_angles = np.arctan2(centers_to_points[:, 1], centers_to_points[:, 0]) % (2 * np.pi)
        start_angle_norm = start_angle % (2 * np.pi)
        end_angle_norm = end_angle % (2 * np.pi)
        
        # Determine if points are within the arc's angular span
        if start_angle_norm <= end_angle_norm:
            within_arc = (points_angles >= start_angle_norm) & (points_angles <= end_angle_norm)
        else:
            within_arc = (points_angles >= start_angle_norm) | (points_angles <= end_angle_norm)

        if if_centers_inside[i-1] == False:
            within_arc = ~ within_arc

        within_ring = np.logical_and(distances_to_center >= arc_center_distances[i], distances_to_center <= radius)
        within_arc = np.logical_and(within_arc, within_ring)
        
        if if_centers_inside[i-1]:
            # XOR with current inside status since circle is "negative space" if center is inside polygon
            inside = np.logical_or(inside, within_arc)
        else:
            inside = np.logical_and(inside, ~within_arc)

    return inside, middle_points_intermideate_line_segments

# from utils_generation import point_to_line_distance
def points_to_line_distance(points, line_start, line_end): # +|
    """Calculate signed distance from point to line segment"""
    line_vec = line_end - line_start
    point_vec = points - line_start
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    point_vec_scaled = point_vec / line_len
    t = np.dot(point_vec_scaled, line_unitvec.reshape(-1, 1)).reshape(-1)
    t = np.clip(t, 0.0, 1.0)
    nearest = line_start + t.reshape(-1, 1) * line_vec
    dist = np.linalg.norm(points - nearest, axis=1)
    return dist

def signed_distance_polygon(points, line_segments, arc_segments, vertices, smooth_factor=40, heaviside=True): # + |
    distances = []
    for line_segment in line_segments:
        distances.append(points_to_line_distance(points, line_segment[0], line_segment[1]))
    for arc_segment in arc_segments:
        distances.append(points_to_arc_distances(points,
                                                arc_segment[0],
                                                arc_segment[1],
                                                arc_segment[2],
                                                arc_segment[3]))
    
    distances = np.array(distances)
    min_distances = np.min(distances, axis=0)

    inside_grid, middle_points_intermideate_line_segments = if_points_in_polygon(points, line_segments, arc_segments, vertices)

    min_distances[~inside_grid] = -min_distances[~inside_grid]
    if heaviside:
        return 1/(1 + np.exp(-smooth_factor*min_distances))
    else:
        return min_distances
    

def SDF_polygon_3D(points, bottom_level, line_segments, arc_segments, vertices, smooth_factor=40):
    """
    Calculate signed distance from points to a 3D polygon
    
    Parameters:
    points: np.array([[x1, y1, z1], [x2, y2, z2], ...]) - points to calculate distance from
    bottom_level: float - bottom level of the polygon (z-axis)
    line_segments: list of tuples - line segments of the polygon
    arc_segments: list of tuples - arc segments of the polygon
    vertices: np.array([[x1, y1, z1], [x2, y2, z2], ...]) - vertices of the polygon
    smooth_factor: float - smooth factor for the signed distance
    """
    
    points_xy = points[:, :2]
    points_z = points[:, 2]

    sdf_xy = signed_distance_polygon(points_xy,
                                     line_segments,
                                     arc_segments,
                                     vertices,
                                     smooth_factor=smooth_factor,
                                     heaviside=False)
    sdf_z = points_z - bottom_level

    in_contour_over_bottom_level = np.logical_and(sdf_xy > 0, sdf_z >= 0)
    in_contour_under_bottom_level = np.logical_and(sdf_xy > 0, sdf_z < 0)
    out_contour_over_bottom_level = np.logical_and(sdf_xy <= 0, sdf_z >= 0)
    out_contour_under_bottom_level = np.logical_and(sdf_xy <= 0, sdf_z < 0)

    sdf_3d = np.zeros_like(sdf_xy)
    sdf_3d[in_contour_over_bottom_level] = np.min([sdf_xy[in_contour_over_bottom_level], sdf_z[in_contour_over_bottom_level]], axis=0)
    sdf_3d[in_contour_under_bottom_level] = sdf_z[in_contour_under_bottom_level]
    sdf_3d[out_contour_over_bottom_level] = sdf_xy[out_contour_over_bottom_level]
    sdf_3d[out_contour_under_bottom_level] = -np.sqrt(sdf_xy[out_contour_under_bottom_level]**2 + sdf_z[out_contour_under_bottom_level]**2)

    return sdf_3d

# Plot a sample from the generated DataFrame
def plot_sample_from_df(df, points_df, sample_index=0):
    # Extract the sdf string and convert it back to a list of floats
    sdf_str = df.iloc[sample_index]['sdf']
    sdf = np.array(list(map(float, sdf_str.split(','))))
    
    # Reshape the sdf array to match the grid shape
    point_per_side = int(np.sqrt(len(sdf)))
    sdf = sdf.reshape((point_per_side, point_per_side))
    
    # Extract the points grid
    x = points_df['x'].values.reshape((point_per_side, point_per_side))
    y = points_df['y'].values.reshape((point_per_side, point_per_side))
    
    # Plot the sdf values
    plt.figure(figsize=(8, 6))
    plt.contourf(x, y, sdf, levels=50, cmap='RdBu')
    plt.colorbar(label='Signed Distance Function (SDF)')
    plt.title('Sample SDF Plot from DataFrame')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

#### Reconstruction utils ########


def get_rounded_polygon(vertices, radiuses): # + ->
    """Get arc segments and line segments for a polygon with rounded corners."""
    
    
    num_vertices = vertices.shape[0]
    line_segments = []
    arc_segments = []
    arc_ends = []
    v_p_distances = []

    first_middle_segments = []
    second_middle_segments = [] 

    # random_variables = np.random.uniform(0.2, 0.8, num_vertices)

    for i in range(num_vertices):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % num_vertices]
        v3 = vertices[(i + 2) % num_vertices]
        radius = radiuses[(i) % num_vertices]

        p1_new, p2_new, center, start_angle, end_angle = get_corner_points(v1, v2, v3, radius)
    
        v_p_distances.append(np.linalg.norm(v2 - p2_new))
        arc_ends.append((p2_new, p1_new))
        arc_segments.append((center, start_angle, end_angle, radius))            

    for i in range(num_vertices):
        line_segments.append((arc_ends[i][0], arc_ends[(i + 1) % num_vertices][1]))
        
    return line_segments, arc_segments, False

def extract_geometry(chi):
    line_width = 2
    
    geometry_label = chi[0]
    geometry_label = round(geometry_label * 2) / 2

    if geometry_label == 0:
        sigmas_ratio = chi[1]*1.5
        a = 0.5
        b = a * sigmas_ratio
        return "ellipse", [sigmas_ratio, a, b]

    else:
        v1 = np.array([-0.5, -0.5])  # First point of the edge parallel to x-axis
        v2 = np.array([0.5, -0.5])   # Second point of the edge parallel to x-axis 
        vertices = [v1, v2]
        radiuses = []

        if geometry_label == 0.5:
            x3 = chi[2]
            y3 = chi[3]
            v3 = np.array([x3, y3])

            vertices.append(v3)

            for i in range(3):
                radiuses.append(chi[i+4])

        elif geometry_label == 1:
            x3 = chi[7]
            y3 = chi[8]
            x4 = chi[9]
            y4 = chi[10]

            v3 = np.array([x3, y3])
            v4 = np.array([x4, y4])

            vertices.append(v3)
            vertices.append(v4)

            for i in range(4):
                radiuses.append(chi[i+11])

        else:
            print("Unknown geometry label: ", geometry_label, chi)
            return None, None

        vertices = np.array(vertices)
        radiuses = np.array(radiuses)*3
        radiuses = np.clip(radiuses, 0.01, None)

        line_segments, arc_segments, arcs_intersection = get_rounded_polygon(vertices, radiuses)

        return "polygon", [vertices, radiuses, line_segments, arc_segments]

#########################################################################

def plot_feature_sdf_item(
        smooth_factor=40,
        num_points=100,
        scatter_cmap='viridis',
        filename=None,
        axes_length=1,
        min_radius=0.1,
        line_width=0.001,
        sdf_threshold_min=0.001,
        sdf_threshold_max=0.999,
        feature_type='triangle',
        text_size=45
):
    from triangle_sdf import generate_triangle
    from quadrangle_sdf import generate_quadrangle
    from ellipse_sdf import ellipse_sdf
    import matplotlib.tri as tri
    from matplotlib.colors import TwoSlopeNorm

    # Create figure and axis
    plt.figure(figsize=(8, 8))
    ax = plt.gca()


    if feature_type == 'triangle':
        while True:
            vertices = generate_triangle()
            line_segments, arc_segments, arcs_intersection = (
                get_rounded_polygon_segments_rand_radius(vertices, min_radius))
            if arcs_intersection == False:
                break

    elif feature_type == 'quadrangle':
        while True:
            vertices = generate_quadrangle()
            line_segments, arc_segments, arcs_intersection = (
                get_rounded_polygon_segments_rand_radius(vertices, 0.1))
            if arcs_intersection == False:
                break

    if feature_type == 'triangle' or feature_type == 'quadrangle':
        perimeter, line_perimeter, arc_perimeter = compute_perimeter(line_segments, arc_segments)
        # arc_ratio = arc_perimeter / perimeter
        arc_radii = np.array([radius for _, _, _, radius in arc_segments])
        vertces_list = []

    if feature_type == 'triangle':
        v1, v2, v3 = vertices
        vertces_list.append(v1)
        vertces_list.append(v2)
        vertces_list.append(v3)
        # Sample more points near the triangle
        feature_center = (v1 + v2 + v3) / 3

    elif feature_type == 'quadrangle':
        v1, v2, v3, v4 = vertices
        vertces_list.append(v1)
        vertces_list.append(v2)
        vertces_list.append(v3)
        vertces_list.append(v4)
        # Sample more points near the quadrangle
        feature_center = (v1 + v2 + v3 + v4) / 4
    else:
        feature_center = np.array([0, 0])
    
    
    # Gaussian sampling around the triangle
    # points_gaussian = np.random.normal(loc=feature_center, scale=0.25, size=(num_points, 2))
    # points_gaussian = np.clip(points_gaussian, -0.8, 0.8)

    # if feature_type == 'triangle' or feature_type == 'quadrangle':
    #     # Generate points in the vertices
    #     points_in_vertices = []
    #     for center, radius in zip(vertces_list, arc_radii):
    #         points_in_circle = np.random.normal(loc=center, scale=0.2, size=(int(num_points/3), 2))
    #         points_in_circle = np.clip(points_in_circle, -1, 1)
    #         points_in_vertices.append(points_in_circle)
        
    #     points_in_vertices = np.vstack(points_in_vertices)

    #     points = np.vstack([points_gaussian, points_in_vertices])
    # else:
    #     points = points_gaussian

    point_per_side = int(np.sqrt(num_points))
    x = np.linspace(-axes_length, axes_length, point_per_side)
    y = np.linspace(-axes_length, axes_length, point_per_side)
    X, Y = np.meshgrid(x, y)
    points = np.array([X.flatten(), Y.flatten()]).T

    # Remove points that are too close to each other

    # min_distance = 0.05  # Minimum distance between points
    # keep_indices = []
    # for i in range(len(points)):
    #     # If this point is already marked for removal, skip it
    #     if i not in keep_indices:
    #         # Calculate distances to all other points
    #         distances = np.linalg.norm(points[i] - points, axis=1)
    #         # Find points that are too close (excluding self)
    #         close_points = np.where(distances < min_distance)[0]
    #         # If this point hasn't been marked for removal yet, keep it and remove others
    #         if not any(j in keep_indices for j in close_points):
    #             keep_indices.append(i)
        
    # points = points[keep_indices]

    if feature_type == 'triangle' or feature_type == 'quadrangle':
        sdf = signed_distance_polygon(points, line_segments, arc_segments, vertices, smooth_factor=smooth_factor)
    else:   
        a = 0.5
        b_w = np.random.uniform(0.5, 1.5)  # Semi-minor axis (smaller than a)
        b = a * b_w
        
        sdf = ellipse_sdf(points, a, b)
        sdf = 1/(1 + np.exp(smooth_factor*sdf))

    points = points[sdf > sdf_threshold_min]
    sdf = sdf[sdf > sdf_threshold_min]

    points = points[sdf < sdf_threshold_max]
    sdf = sdf[sdf < sdf_threshold_max]

    # Plot points colored by SDF using pcolormesh
    # x_unique = np.unique(points[:, 0])
    # y_unique = np.unique(points[:, 1])
    # X, Y = np.meshgrid(x_unique, y_unique)
    # Z = sdf.reshape(len(y_unique), len(x_unique))
    # Create triangulation from points
    # Create triangulation with max edge length constraint to avoid large triangles
    triang = tri.Triangulation(points[:, 0], points[:, 1])
    # Remove triangles with edges longer than threshold
    max_edge_length = 0.1  # Adjust this threshold as needed
    triangles = triang.triangles
    x = triang.x
    y = triang.y
    mask = np.ones(len(triangles), dtype=bool)
    
    for i, triangle in enumerate(triangles):
        # Get vertices of triangle
        verts = np.column_stack((x[triangle], y[triangle]))
        # Calculate edge lengths
        edges = np.roll(verts, -1, axis=0) - verts
        edge_lengths = np.sqrt(np.sum(edges**2, axis=1))
        # Mask triangles with long edges
        if np.any(edge_lengths > max_edge_length):
            mask[i] = False
            
    triang.set_mask(~mask)
    mesh = ax.tripcolor(triang, sdf, cmap=scatter_cmap, shading='gouraud')
    # 'PiYG'

    if feature_type == 'triangle' or feature_type == 'quadrangle':
        # Plot dashed lines between vertices
        # Extract x and y coordinates from vertices
        x_coords = [v[0] for v in vertices]
        y_coords = [v[1] for v in vertices]
        # Add first vertex again to close the polygon
        x_coords.append(vertices[0][0])
        y_coords.append(vertices[0][1])
        ax.plot(x_coords, y_coords, 'k--', alpha=0.5)
    
    # Add coordinate axes
    ax.arrow(x=-axes_length*0.75, y=0, dx=2*axes_length*0.75, dy=0, color='k', alpha=0.3, linewidth=1.5, head_width=0.05, head_length=0.05, length_includes_head=True)
    ax.arrow(x=0, y=-axes_length*0.75, dx=0, dy=2*axes_length*0.75, color='k', alpha=0.3, linewidth=1.5, head_width=0.05, head_length=0.05, length_includes_head=True)

    if feature_type == 'triangle' or feature_type == 'quadrangle':
        # Plot vertices and labels
        v_x = [v[0] for v in vertices]
        v_y = [v[1] for v in vertices]
        ax.scatter(v_x, v_y, color='black', marker='+', s=600, zorder=50, linewidth=6)
        ax.scatter(v_x, v_y, color='darkgreen', marker='+', s=400, zorder=50, linewidth=3)

        if feature_type == 'triangle':
            ax.text(v1[0]+0.035, v1[0]-0.035, '$v_1, R_1$', fontsize=text_size, ha='left', va='top', bbox=dict(facecolor='white', edgecolor='black', alpha=0.9, linewidth=2))
            ax.text(v2[0]-0.04, v2[1]-0.035, '$v_2, R_2$', fontsize=text_size, ha='right', va='top', bbox=dict(facecolor='white', edgecolor='black', alpha=0.9, linewidth=2))
            ax.text(v3[0]-0.035, v3[1]+0.035, '$v_3, R_3$', fontsize=text_size, ha='left', va='bottom', bbox=dict(facecolor='white', edgecolor='black', alpha=0.9, linewidth=2))
           
        if feature_type == 'quadrangle':
            ax.text(v1[0]+0.035, v1[0]-0.035, '$v_4, R_4$', fontsize=text_size, ha='left', va='top', bbox=dict(facecolor='white', edgecolor='black', alpha=0.9, linewidth=2))
            ax.text(v2[0]-0.04, v2[1]-0.035, '$v_5, R_5$', fontsize=text_size, ha='right', va='top', bbox=dict(facecolor='white', edgecolor='black', alpha=0.9, linewidth=2))
            ax.text(v3[0]-0.035, v3[1]+0.035, '$v_6, R_6$', fontsize=text_size, ha='left', va='bottom', bbox=dict(facecolor='white', edgecolor='black', alpha=0.9, linewidth=2))
            ax.text(v4[0]+0.04, v4[1]+0.035, '$v_7, R_7$', fontsize=text_size, ha='left', va='bottom', bbox=dict(facecolor='white', edgecolor='black', alpha=0.9, linewidth=2))

        # Plot line segments
        for start, end in line_segments:
            ax.plot([start[0], end[0]], [start[1], end[1]], 'g-', label='Line segments', linewidth=line_width)

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
            ax.plot(x, y, 'r-', label='Arc segments', linewidth=line_width)

    if feature_type == 'ellipse':
        # Create ellipse patch
        from matplotlib.patches import Ellipse
        ellipse = Ellipse(np.array([0, 0]), 2*a, 2*b, fill=False, color='r', linewidth=line_width)
        
        ax.add_patch(ellipse)

        # Add dimension lines for semi-major and semi-minor axes
        # Semi-major axis (a)
        ax.plot([0, a+0.03], [-b, -b], 'k:', linewidth=1.5)
        # ax.plot([-a, 0], [0, 0], 'k:', linewidth=1.5)
        # ax.text(a/2, 0.05, 'a', fontsize=20, ha='center', va='bottom')
        
        # Semi-minor axis (b) 
        ax.annotate('', xy=(a+0.03, -b), xytext=(a+0.03, 0), arrowprops=dict(arrowstyle='<->', linestyle='--', linewidth=2.5, color='k', mutation_scale=25))
        # ax.plot([0, 0], [-b, 0], 'k:', linewidth=1.5)
        # ax.text(a, - b/2, 'b', fontsize=20, ha='left', va='center')
        label_b = ax.text(a+0.065, - b/2, 'b', fontsize=text_size, ha='left', va='center', bbox=dict(facecolor='white', edgecolor='black', alpha=0.9, linewidth=2))
        # label_b.set_bbox(dict(facecolor='white', edgecolor='black', alpha=0.9, linewidth=2))
        
        # Add small ticks at ends
        # tick_size = 0.02
        # a-axis ticks
        # ax.plot([a, a], [-tick_size, tick_size], 'k-', linewidth=1.5)
        # ax.plot([-a, -a], [-tick_size, tick_size], 'k-', linewidth=1.5)
        # # b-axis ticks
        # ax.plot([-tick_size, tick_size], [b, b], 'k-', linewidth=1.5)
        # ax.plot([-tick_size, tick_size], [-b, -b], 'k-', linewidth=1.5)


    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(-axes_length*0.8, axes_length*0.8)
    ax.set_ylim(-axes_length*0.8, axes_length*0.8)

    # Remove frame and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_linestyle((0, (10, 10)))  # 5 points on, 5 points off
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linestyle((0, (10, 10)))
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linestyle((0, (10, 10)))
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linestyle((0, (10, 10)))
    ax.spines['left'].set_linewidth(2)

    # Remove duplicate labels
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys())

    # plt.title('Triangle with Rounded Corners')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.05)
    plt.show()


#################################################

import matplotlib.tri as tri
from matplotlib.colors import TwoSlopeNorm

def plot_sdf_heav_item_by_tensor(
        vertices,
        radiuses,
        heaviside, # N x WH
        points, # N x W x H x 2
        filename=None,
        # line_segments,
        # arc_segments        
):
    
    # # Create figure and axis
    # plt.figure(figsize=(8, 8))
    # ax = plt.gca()
    num_samples = points.shape[0]

    # TODO: add line segments and arc segments
    line_segments, arc_segments, arcs_intersection = (
        get_rounded_polygon_segments_with_given_radiuses(vertices, radiuses))

    vertces_list = []

    v1, v2, v3, v4 = vertices
    vertces_list.append(v1)
    vertces_list.append(v2)
    vertces_list.append(v3)
    vertces_list.append(v4)

    fig = plt.figure(figsize=(4*num_samples, 4))

    for i in range(num_samples):
        # First subplot: 2D Feature Contour
        ax1 = fig.add_subplot(1, num_samples, i+1)
        # Second subplot: 3D Surface of SDF

        triang1 = tri.Triangulation(points[i, :, 0], points[i, :, 1])
        norm = TwoSlopeNorm(vcenter=0.5, vmin=heaviside.min(), vmax=heaviside.max())
        mesh1 = ax1.tripcolor(triang1, heaviside[i], cmap='seismic', shading='gouraud', norm=norm)

        num_points = points.shape[1]
        point_per_side = int(np.sqrt(num_points))

        # Plot line segments
        for start, end in line_segments:
            ax1.plot([start[0], end[0]], [start[1], end[1]], 'g-', linewidth=3)
            
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
            ax1.plot(x, y, 'r-', linewidth=3)
            # ax2.plot(x, y, np.zeros_like(x)+z_offset, 'r-', linewidth=line_width)

        # Set equal aspect ratio and limits for 2D contour
        ax1.set_aspect('equal')
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)

        ax1.set_xticks([])
        ax1.set_yticks([])

        for spine in ax1.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.05)
    plt.show()


def plot_sdf_heav_item(
        smooth_factor=40,
        num_points=100,
        scatter_cmap='viridis',
        filename=None,
        axes_length=1,
        min_radius=0.1,
        line_width=0.001,
        sdf_threshold_min=0.001,
        sdf_threshold_max=0.999,
        feature_type='triangle',
        text_size=45,
        azimuth=75,
        elev=21
):
    
    # # Create figure and axis
    # plt.figure(figsize=(8, 8))
    # ax = plt.gca()


    if feature_type == 'triangle':
        while True:
            vertices = generate_triangle()
            line_segments, arc_segments, arcs_intersection = (
                get_rounded_polygon_segments_rand_radius(vertices, min_radius))
            if arcs_intersection == False:
                break

    elif feature_type == 'quadrangle':
        while True:
            vertices = generate_quadrangle()
            line_segments, arc_segments, arcs_intersection = (
                get_rounded_polygon_segments_rand_radius(vertices, 0.1))
            if arcs_intersection == False:
                break

    if feature_type == 'triangle' or feature_type == 'quadrangle':
        perimeter, line_perimeter, arc_perimeter = compute_perimeter(line_segments, arc_segments)
        # arc_ratio = arc_perimeter / perimeter
        arc_radii = np.array([radius for _, _, _, radius in arc_segments])
        vertces_list = []

    if feature_type == 'triangle':
        v1, v2, v3 = vertices
        vertces_list.append(v1)
        vertces_list.append(v2)
        vertces_list.append(v3)
        # Sample more points near the triangle
        feature_center = (v1 + v2 + v3) / 3

    elif feature_type == 'quadrangle':
        v1, v2, v3, v4 = vertices
        vertces_list.append(v1)
        vertces_list.append(v2)
        vertces_list.append(v3)
        vertces_list.append(v4)
        # Sample more points near the quadrangle
        feature_center = (v1 + v2 + v3 + v4) / 4
    else:
        feature_center = np.array([0, 0])
    


    point_per_side = int(np.sqrt(num_points))
    x = np.linspace(-axes_length, axes_length, point_per_side)
    y = np.linspace(-axes_length, axes_length, point_per_side)
    X, Y = np.meshgrid(x, y)
    points = np.array([X.flatten(), Y.flatten()]).T


    if feature_type == 'triangle' or feature_type == 'quadrangle':
        sdf = signed_distance_polygon(points, line_segments, arc_segments, vertices, smooth_factor=smooth_factor, heaviside=False)
        heaviside = signed_distance_polygon(points, line_segments, arc_segments, vertices, smooth_factor=smooth_factor, heaviside=True)
    else:   
        a = 0.5
        b_w = np.random.uniform(0.5, 1.5)  # Semi-minor axis (smaller than a)
        b = a * b_w
        
        sdf = ellipse_sdf(points, a, b)
        heaviside = 1/(1 + np.exp(smooth_factor*sdf))

    fig = plt.figure(figsize=(12, 4))

    # First subplot: 2D Feature Contour
    ax1 = fig.add_subplot(1, 3, 1)
    # Second subplot: 3D Surface of SDF
    from mpl_toolkits.mplot3d import Axes3D  # Ensure 3D plotting is available
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')

    # azimuth, elev = 75, 21
    ax2.view_init(elev, azimuth)

    triang1 = tri.Triangulation(points[:, 0], points[:, 1])
    # Adjust the colormap to center around 0
    norm = TwoSlopeNorm(vcenter=0, vmin=sdf.min(), vmax=sdf.max())
    mesh1 = ax1.tripcolor(triang1, sdf, cmap=scatter_cmap, shading='gouraud', norm=norm)

    X = points[:, 0].reshape((point_per_side, point_per_side))
    Y = points[:, 1].reshape((point_per_side, point_per_side))
    Z = sdf.reshape((point_per_side, point_per_side))
    # Adjust the colormap to center around 0
    z_offset = 0.05

    if feature_type == 'triangle' or feature_type == 'quadrangle':

        # Plot line segments
        for start, end in line_segments:
            ax1.plot([start[0], end[0]], [start[1], end[1]], 'g-', linewidth=line_width)
            # ax2.plot([start[0], end[0]], [start[1], end[1]], [z_offset, z_offset], 'g-', linewidth=line_width)

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
            ax1.plot(x, y, 'r-', linewidth=line_width)
            # ax2.plot(x, y, np.zeros_like(x)+z_offset, 'r-', linewidth=line_width)

        # Plot the surface after plotting the segments to ensure they are not obscured
        surf1 = ax2.plot_surface(X, Y, Z, cmap=scatter_cmap, edgecolor='none', alpha=1, norm=norm)

            
    if feature_type == 'ellipse':
        # Create ellipse patch
        from matplotlib.patches import Ellipse
        ellipse = Ellipse(np.array([0, 0]), 2*a, 2*b, fill=False, color='r', linewidth=line_width)
        
        ax1.add_patch(ellipse)

        # Add dimension lines for semi-major and semi-minor axes
        # Semi-major axis (a)
        ax1.plot([0, a+0.03], [-b, -b], 'k:', linewidth=1.5)
        
        # Semi-minor axis (b) 
        ax1.annotate('', xy=(a+0.03, -b), xytext=(a+0.03, 0), arrowprops=dict(arrowstyle='<->', linestyle='--', linewidth=2.5, color='k', mutation_scale=25))
        label_b = ax1.text(a+0.065, - b/2, 'b', fontsize=text_size, ha='left', va='center', bbox=dict(facecolor='white', edgecolor='black', alpha=0.9, linewidth=2))

    # Set equal aspect ratio and limits for 2D contour
    ax1.set_aspect('equal')
    ax1.set_xlim(-axes_length, axes_length)
    ax1.set_ylim(-axes_length, axes_length)
    ax1.set_title('Feature Contour', fontsize=text_size)

    # Remove frame and ticks for 2D contour
    # ax1.set_xticks([])
    # ax1.set_yticks([])
    # ax1.spines['top'].set_visible(False)
    # ax1.spines['right'].set_visible(False)
    # ax1.spines['bottom'].set_visible(False)
    # ax1.spines['left'].set_visible(False)

    
    
    ax2.set_title('SDF Surface', fontsize=text_size)
    ax2.set_xlabel('X', fontsize=8)
    ax2.set_ylabel('Y', fontsize=8)
    # ax2.set_zlabel('SDF', fontsize=8)
    ax2.set_xlim(-axes_length, axes_length)
    ax2.set_ylim(-axes_length, axes_length)
    ax2.set_xticks(np.linspace(-axes_length, axes_length, 5))  # Make ticks more rare
    ax2.set_yticks(np.linspace(-axes_length, axes_length, 5))  # Make ticks more rare

    # Third subplot: 3D Surface with Heaviside
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.view_init(elev, azimuth)
    H = heaviside.reshape((point_per_side, point_per_side))
    surf2 = ax3.plot_surface(X, Y, H, cmap=scatter_cmap, edgecolor='none', alpha=0.8)
    ax3.set_title('Heaviside Surface', fontsize=text_size)
    ax3.set_xlabel('X', fontsize=8)
    ax3.set_ylabel('Y', fontsize=8)
    # ax3.set_zlabel('Heaviside', fontsize=8)
    ax3.set_xlim(-axes_length, axes_length)
    ax3.set_ylim(-axes_length, axes_length)
    ax3.set_xticks(np.linspace(-axes_length, axes_length, 5))  # Make ticks more rare
    ax3.set_yticks(np.linspace(-axes_length, axes_length, 5))  # Make ticks more rare

    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.05)
    plt.show()
