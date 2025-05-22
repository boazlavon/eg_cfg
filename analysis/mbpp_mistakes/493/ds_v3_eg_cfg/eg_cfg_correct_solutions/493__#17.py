def calculate_polygons(center_x, center_y, size_x, size_y, num_sides):
    polygons = []
    angle_step = 2 * math.pi / num_sides
    offset_x = size_x * 1.5
    offset_y = size_y * math.sqrt(3)
    rows = int(math.ceil(size_y / 2.0))
    cols = int(math.ceil(size_x / 2.0))
    for row in range(-rows, rows + 1):
        for col in range(-cols, cols + 1):
            hex_center_x = center_x + col * offset_x
            hex_center_y = center_y + row * offset_y
            if col % 2 != 0:
                hex_center_y += offset_y / 2.0
            polygon = []
            for i in range(num_sides + 1):
                angle = i * angle_step - math.pi / 6
                x = hex_center_x + size_x * math.cos(angle)
                y = hex_center_y + size_y * math.sin(angle)
                point = (x, y)
                polygon.append(point)
            polygons.append(polygon)
    return polygons