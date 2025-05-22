import math

def calculate_polygons(origin_x, origin_y, size_x, size_y, side_length):
    polygons = []
    angle = 2 * math.pi / 6  # 60 degrees in radians
    hex_radius = side_length / math.sin(angle)
    
    for i in range(size_x):
        for j in range(size_y):
            # Calculate center coordinates
            center_x = origin_x + 1.5 * side_length * i
            center_y = origin_y + math.sqrt(3) * side_length * (j - 0.5 * (i % 2))
            
            # Calculate vertices
            vertices = []
            for k in range(6):
                vertex_x = center_x + hex_radius * math.cos(angle * k)
                vertex_y = center_y + hex_radius * math.sin(angle * k)
                vertices.append((vertex_x, vertex_y))
            
            # Close the polygon by adding the first vertex again
            vertices.append(vertices[0])
            polygons.append(vertices)
    
    return polygons