def calculate_polygons(center_x, center_y, size_x, size_y, sides):
    polygons = []
    radius = size_x / 2
    angle_step = 2 * math.pi / sides
    start_angle = -math.pi / 2
    apothem = radius * math.cos(math.pi / sides)
    horizontal_spacing = 3 * radius
    vertical_spacing = 2 * apothem
    for row in range(size_y):
        for col in range(size_x):
            current_x = center_x + col * horizontal_spacing
            current_y = center_y + row * vertical_spacing
            if col % 2 != 0:
                current_y += apothem
            polygon = []
            for i in range(sides + 1):
                angle = start_angle + i * angle_step
                x = current_x + radius * math.cos(angle)
                y = current_y + radius * math.sin(angle)
                point = (x, y)
                polygon.append(point)
            polygons.append(polygon)
    return polygons