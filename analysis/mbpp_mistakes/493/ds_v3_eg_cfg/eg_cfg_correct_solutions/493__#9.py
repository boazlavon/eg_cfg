import math

def calculate_polygons(center_x, center_y, outer_radius, cols, rows):
    angle = 2 * math.pi / 6
    polygons = []
    row_radius = outer_radius * math.sqrt(3) / 2
    horizontal_step = 1.5 * outer_radius
    vertical_step = row_radius * 2
    start_x = center_x - (cols - 1) * horizontal_step / 2
    start_y = center_y - (rows - 1) * row_radius
    for row in range(rows):
        offset_x = 0 if row % 2 == 0 else horizontal_step / 2
        current_row_y = start_y + row * vertical_step
        for col in range(cols):
            current_center_x = start_x + offset_x + col * horizontal_step
            hexagon = []
            for i in range(6):
                point_angle = angle * (i + 0.5)
                x_point = current_center_x + outer_radius * math.cos(point_angle)
                y_point = current_row_y + outer_radius * math.sin(point_angle)
                point_tuple = (x_point, y_point)
                hexagon.append(point_tuple)
            first_point_copy = hexagon[0]
            polygon = hexagon + [first_point_copy]
            polygons.append(polygon)
    return polygons