import math

def calculate_polygons(origin_x, origin_y, width, height, n):
    hexagons = []
    radius = width / 4
    side_length = radius * math.sqrt(3)
    for j in range(n):
        for i in range(n):
            if j % 2 == 0:
                center_x = origin_x + i * (3 * radius)
            else:
                center_x = origin_x + (i + 0.5) * (3 * radius)
            center_y = origin_y + j * (side_length * 1.5)
            hexagon = []
            for k in range(6):
                angle_deg = 60 * k + 30
                angle_rad = math.pi * angle_deg / 180
                point_x = center_x + radius * math.cos(angle_rad)
                point_y = center_y + radius * math.sin(angle_rad)
                point = (point_x, point_y)
                hexagon.append(point)
            hexagon.append(hexagon[0])
            hexagons.append(hexagon)
    return hexagons