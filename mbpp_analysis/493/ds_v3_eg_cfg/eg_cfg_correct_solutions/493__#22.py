import math

def calculate_polygons(offset_x, offset_y, size_x, size_y, n):
    polygons = []
    for i in range(size_x):
        for j in range(size_y):
            center_x = offset_x + i * 3 * n
            center_y = offset_y + j * math.sqrt(3) * n
            if j % 2 == 1:
                center_x += 1.5 * n
            hexagon = []
            for k in range(6):
                angle_deg = 60 * k + 30
                angle_rad = math.pi * angle_deg / 180
                x = center_x + n * math.cos(angle_rad)
                y = center_y + n * math.sin(angle_rad)
                hexagon.append((x, y))
            hexagon.append(hexagon[0])  # Close the hexagon by repeating the first point
            polygons.append(hexagon)
    return polygons