import math

def calculate_polygons(origin_x, origin_y, radius, cols, rows):
    polygons = []
    side_length = radius
    angle = math.pi / 3  # 60 degrees in radians
    
    for row in range(rows):
        for col in range(cols):
            # Calculate the center of the hexagon
            if row % 2 == 0:
                center_x = origin_x + col * (2 * side_length * math.sin(angle))
            else:
                center_x = origin_x + col * (2 * side_length * math.sin(angle)) + side_length * math.sin(angle)
            center_y = origin_y + row * (1.5 * side_length)
            
            # Calculate the 6 vertices of the hexagon
            vertices = []
            for i in range(6):
                x = center_x + side_length * math.cos(i * angle)
                y = center_y + side_length * math.sin(i * angle)
                vertices.append((round(x, 12), round(y, 12)))
            
            # The first vertex is repeated to close the polygon
            polygon = vertices + [vertices[0]]
            polygons.append(polygon)
    
    return polygons