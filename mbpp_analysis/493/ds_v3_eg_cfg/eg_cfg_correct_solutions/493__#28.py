import math

def calculate_polygons(center_x, center_y, size, rows, cols):
    polygons = []
    angle_step = 2 * math.pi / 6
    for row in range(rows):
        for col in range(cols):
            hex_center_x = center_x + col * (size * 1.5)
            hex_center_y = center_y + row * (size * math.sqrt(3))
            if row % 2 == 1:
                hex_center_x += size * 0.75
            vertices = []
            for i in range(6):
                angle = angle_step * i - math.pi / 6
                x = hex_center_x + size * math.cos(angle)
                y = hex_center_y + size * math.sin(angle)
                vertices.append((x, y))
            vertices.append(vertices[0])
            polygons.append(vertices)
    return polygons