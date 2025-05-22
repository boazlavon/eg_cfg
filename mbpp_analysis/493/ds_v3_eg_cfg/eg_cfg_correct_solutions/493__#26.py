import math

def calculate_polygons(origin_x, origin_y, radius, num_rows, num_cols):
    polygons = []
    sqrt3 = math.sqrt(3)
    hex_height = radius * sqrt3
    hex_width = radius * 2
    half_width = hex_width / 2
    half_height = hex_height / 2
    for row in range(num_rows):
        row_y = origin_y + row * hex_height
        row_offset = 0.5 if row % 2 else 0
        for col in range(num_cols):
            center_x = origin_x + col * hex_width + row_offset * hex_width
            center_y = row_y
            polygon = []
            for i in range(7):
                angle_deg = 60 * i - 30
                angle_rad = math.pi / 180 * angle_deg
                point_x = center_x + radius * math.cos(angle_rad)
                point_y = center_y + radius * math.sin(angle_rad)
                point = (point_x, point_y)
                polygon.append(point)
            polygons.append(polygon)
    return polygons