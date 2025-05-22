def calculate_polygons(center_x, center_y, size, grid_width, grid_height):
    hexagons = []
    angle = 2 * math.pi / 6
    for row in range(grid_height):
        for col in range(grid_width):
            offset_x = col * size * 3 / 2
            offset_y = row * size * math.sqrt(3)
            if row % 2 == 1:
                offset_x += size * 3 / 4
            vertices = []
            for i in range(6):
                x = center_x + offset_x + size * math.cos(i * angle)
                y = center_y + offset_y + size * math.sin(i * angle)
                x_rounded = round(x, 15)
                y_rounded = round(y, 15)
                vertices.append((x_rounded, y_rounded))
            vertices.append(vertices[0])
            hexagons.append(vertices)
    return hexagons