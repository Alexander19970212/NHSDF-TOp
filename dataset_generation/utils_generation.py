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


def get_rounded_polygon_segments_rand_radius(vertices, min_radius = 0.1): # + ->
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
            
            valid_radius = np.random.uniform(min_radius, max_radius)
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

def signed_distance_polygon(points, line_segments, arc_segments, vertices): # + |
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
    return 1/(1 + np.exp(-40*min_distances))

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