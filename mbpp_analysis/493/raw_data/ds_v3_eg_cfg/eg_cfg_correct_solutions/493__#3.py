import math

def calculate_polygons(center_x, center_y, size, rows, cols):
    polygons = []
    angle = 2 * math.pi / 6
    for row in range(rows):
        for col in range(cols):
            polygon = []
            x_offset = col * size * 1.5
            y_offset = row * size * math.sqrt(3)
            if col % 2 != 0:
                y_offset += size * math.sqrt(3) / 2
            for i in range(6):
                x = center_x + x_offset + size * math.cos(i * angle)
                y = center_y + y_offset + size * math.sin(i * angle)
                polygon.append((x, y))
            polygon.append(polygon[0])  # Close the polygon by repeating the first point
            polygons.append(polygon)
    return polygons