import math

def calculate_polygons(center_x, center_y, size_x, size_y, num_sides):
    polygons = []
    angle = 2 * math.pi / num_sides
    radius = size_x
    vertical_step = size_y * 1.5
    horizontal_step = size_x * math.sqrt(3)
    half_width = size_x * 0.5
    half_height = size_y * 0.5 * math.sqrt(3)
    for row in range(num_sides):
        for col in range(num_sides):
            x = center_x + col * horizontal_step - row * half_width
            y = center_y + row * vertical_step - col * half_height
            polygon = []
            for i in range(num_sides + 1):
                current_angle = i * angle
                dx = radius * math.sin(current_angle)
                dy = radius * math.cos(current_angle)
                point_x = x + dx
                point_y = y + dy
                point = (point_x, point_y)
                polygon.append(point)
            polygons.append(polygon)
    return polygons