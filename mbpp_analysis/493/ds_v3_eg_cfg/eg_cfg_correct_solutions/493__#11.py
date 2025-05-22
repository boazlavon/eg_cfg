def calculate_polygons(center_x, center_y, size_x, size_y, offset):
    import math
    h = size_x * math.sqrt(3) / 2
    full_set = []
    current_x = center_x - (offset - 1) * size_x * 3 / 2
    for i in range(0, offset * 2):
        current_y = center_y - (offset - 1) * 2 * h - i % 2 * h
        for j in range(0, offset):
            polygon = []
            x_left_side = current_x - size_x
            if (i + j) < offset:
                y_first_point_top = (j + offset - 1) * 2 * h + i % 2 * h
                adjusted_center_y_offset = size_y / 2
            elif (i + j) > (3 * offset - 2):
                bottom_adjustment = offset * 2 * h * (j - offset)
                y_first_point_top = current_y + bottom_adjustment
                adjusted_center_y_offset = size_y / 6
            else:
                y_first_point_top = current_y + offset * size_y - j * h
                adjusted_center_y_offset = size_y / 4
            x1_current = x_left_side - adjusted_center_y_offset
            y1_first = y_first_point_top - adjusted_center_y_offset
            polygon_point1 = (x1_current, y1_first)
            x2_current = x_left_side + size_x / 2 - adjusted_center_y_offset
            y2_first = y1_first - adjusted_center_y_offset + math.sqrt(3) / 2
            polygon_point2 = (x2_current, y2_first)
            x3_current = x_left_side + size_x - adjusted_center_y_offset
            y3_first = y1_first - adjusted_center_y_offset
            polygon_point3 = (x3_current, y3_first)
            x4_current = x3_current
            y4_first_part = y3_first + size_y - adjusted_center_y_offset * 2
            polygon_point4 = (x4_current, y4_first_part)
            x5_current = x2_current
            y5_first_part = y4_first_part + math.sqrt(3) / 2 * 2 + adjusted_center_y_offset * 3