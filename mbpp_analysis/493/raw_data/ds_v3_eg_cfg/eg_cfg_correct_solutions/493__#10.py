import math

def calculate_polygons(center_x, center_y, size, rows, cols):
    polygons = []
    sqrt3 = math.sqrt(3)
    for row in range(rows):
        for col in range(cols):
            x = center_x + col * 1.5 * size
            y = center_y + row * sqrt3 * size
            if col % 2 != 0:
                y += sqrt3 * size / 2
            hexagon = []
            for i in range(6):
                angle_deg = 60 * i - 30
                angle_rad = math.radians(angle_deg)
                x_pt = x + size * math.cos(angle_rad)
                y_pt = y + size * math.sin(angle_rad)
                hexagon.append((x_pt, y_pt))
            hexagon.append(hexagon[0])  # Close the hexagon by repeating the first point
            polygons.append(hexagon)
    return polygons