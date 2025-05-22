import math

def calculate_polygons(origin_x, origin_y, radius, horizontal_count, vertical_count):
    polygons = []
    sqrt3 = math.sqrt(3)
    horizontal_spacing = radius * 3
    vertical_spacing = radius * sqrt3
    for row in range(vertical_count):
        for col in range(horizontal_count):
            center_x = origin_x + col * horizontal_spacing
            center_y = origin_y + row * vertical_spacing
            if row % 2 == 1:
                center_x += horizontal_spacing / 2
            points = []
            for i in range(6):
                angle_deg = 60 * i - 30
                angle_rad = math.radians(angle_deg)
                x = center_x + radius * math.cos(angle_rad)
                y = center_y + radius * math.sin(angle_rad)
                point = (x, y)
                points.append(point)
            closing_point = points[0]
            points.append(closing_point)
            polygons.append(points)
    return polygons