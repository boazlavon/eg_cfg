import math

def calculate_polygons(center_x, center_y, size_x, size_y, sides):
    angle = 2 * math.pi / sides
    polygons = []
    for i in range(size_x):
        row = center_x + i * 4  # Adjust the spacing as needed
        for j in range(size_y):
            col_offset = 3.464 * (j % 2) if sides % 2 == 0 else 0  # Stagger rows for even-sided polygons
            col = center_y + j * 3.464  # Adjust the vertical spacing as needed
            polygon = []
            for k in range(sides + 1):  # Include an extra point to close the polygon
                x = row + col_offset + math.cos(k * angle - math.pi / 2) * 2  # Radius of 2
                y = col + math.sin(k * angle - math.pi / 2) * 2  # Radius of 2
                polygon.append((round(x, 12), round(y, 12)))  # Round to 12 decimal places
            polygons.append(polygon)
    return polygons