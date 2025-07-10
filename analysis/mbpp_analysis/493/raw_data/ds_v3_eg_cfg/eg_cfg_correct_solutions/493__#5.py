def calculate_polygons(offset_x, offset_y, side_length, num_rows, num_cols):
    polygons = []
    sqrt3 = math.sqrt(3)
    for row in range(num_rows):
        for col in range(num_cols):
            if row % 2 == 0:
                center_x = offset_x + 1.5 * side_length * col
            else:
                center_x = offset_x + 1.5 * side_length * col + 0.75 * side_length
            center_y = offset_y - row * side_length * sqrt3 / 2
            polygon = []
            for i in range(6):
                angle_deg = 60 * i + 30
                angle_rad = math.radians(angle_deg)
                x = center_x + side_length * math.cos(angle_rad)
                y = center_y + side_length * math.sin(angle_rad)
                polygon.append((x, y))
            polygon.append(polygon[0])
            polygons.append(polygon)
    return polygons