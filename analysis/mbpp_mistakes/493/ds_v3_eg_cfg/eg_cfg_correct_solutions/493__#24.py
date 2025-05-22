import math

def calculate_polygons(origin_x, origin_y, radius, cols, rows):
    polygons = []
    width = radius * 3 / 2
    height = radius * math.sqrt(3)
    for row in range(rows):
        for col in range(cols):
            center_x = origin_x + col * width
            if row % 2 == 1:
                center_x += width / 2
            center_y = origin_y + row * height
            hexagon = []
            for i in range(6):
                angle_deg = 60 * i - 30
                angle_rad = math.pi / 180 * angle_deg
                x = center_x + radius * math.cos(angle_rad)
                y = center_y + radius * math.sin(angle_rad)
                hexagon.append((x, y))
            hexagon.append(hexagon[0])
            polygons.append(hexagon)
    return polygons