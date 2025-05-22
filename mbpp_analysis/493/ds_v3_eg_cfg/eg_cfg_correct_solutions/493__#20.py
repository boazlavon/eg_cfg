import math

def calculate_polygons(origin_x, origin_y, width, height, n):
    polygons = []
    side_length = width / n
    radius = side_length / math.sin(math.pi / 3)
    vertical_step = radius * 1.5
    horizontal_step = side_length * math.sqrt(3)
    rows = int(height / vertical_step) + 2
    cols = int(width / horizontal_step) + 2
    for row in range(rows):
        for col in range(cols):
            center_x = origin_x + col * horizontal_step
            if row % 2 != 0:
                center_x = center_x + horizontal_step / 2
            center_y = origin_y + row * vertical_step
            polygon = []
            for i in range(6):
                angle_deg = 60 * i - 30
                angle_rad = math.pi / 180 * angle_deg
                x = center_x + radius * math.cos(angle_rad)
                y = center_y + radius * math.sin(angle_rad)
                point = (x, y)
                polygon.append(point)
            polygon.append(polygon[0])
            polygons.append(polygon)
    return polygons