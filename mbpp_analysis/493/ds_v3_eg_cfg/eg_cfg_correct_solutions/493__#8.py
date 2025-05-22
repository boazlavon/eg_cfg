import math

def calculate_polygons(offset_x, offset_y, size_x, size_y, radius):
    polygons = []
    sqrt_3 = math.sqrt(3)
    hex_width = radius * 2
    hex_height = sqrt_3 * radius
    x_step = 1.5 * radius
    y_step = sqrt_3 * radius
    
    for row in range(size_y):
        for col in range(size_x):
            center_x = offset_x + col * x_step * 2
            if row % 2 != 0:
                center_x = center_x + x_step
            center_y = offset_y + row * y_step
            
            polygon = []
            for i in range(6):
                angle_deg = 60 * i - 30
                angle_rad = math.pi / 180 * angle_deg
                point_x = center_x + radius * math.cos(angle_rad)
                point_y = center_y + radius * math.sin(angle_rad)
                point = (point_x, point_y)
                polygon.append(point)
            
            closing_point = polygon[0]
            polygon.append(closing_point)
            polygons.append(polygon)
    
    return polygons