import math

def calculate_polygons(center_x, center_y, size, num_columns, num_rows):
    polygons = []
    sqrt3 = math.sqrt(3)
    for row in range(num_rows):
        for col in range(num_columns):
            x = center_x + col * 1.5 * size
            y = center_y + row * sqrt3 * size
            if col % 2 != 0:
                y += sqrt3 * size / 2
            hexagon = []
            for i in range(6):
                angle_deg = 60 * i
                angle_rad = math.pi / 180 * angle_deg
                hexagon_x = x + size * math.cos(angle_rad)
                hexagon_y = y + size * math.sin(angle_rad)
                hexagon.append((hexagon_x, hexagon_y))
            polygons.append(hexagon + [hexagon[0]])
    return polygons