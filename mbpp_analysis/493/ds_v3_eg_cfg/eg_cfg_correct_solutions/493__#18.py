import math

def calculate_polygons(center_x, center_y, side_length, rows, columns):
    polygons = []
    angle = 2 * math.pi / 6
    offset_y = side_length * math.sqrt(3)

    hexagon_radius = side_length
    vertical_spacing = offset_y
    horizontal_spacing = hexagon_radius * 1.5

    start_x = center_x - (columns - 1) * horizontal_spacing / 2
    start_y = center_y - (rows - 1) * vertical_spacing / 2

    for row in range(rows):
        row_offset = row % 2
        row_y = start_y + row * vertical_spacing

        for col in range(columns):
            col_x = start_x + col * horizontal_spacing + row_offset * horizontal_spacing / 2
            hexagon_points = []

            for i in range(7):
                angle_i = angle * i - math.pi / 6
                point_x = col_x + hexagon_radius * math.cos(angle_i)
                point_y = row_y + hexagon_radius * math.sin(angle_i)
                hexagon_points.append((point_x, point_y))

            polygons.append(hexagon_points)

    return polygons