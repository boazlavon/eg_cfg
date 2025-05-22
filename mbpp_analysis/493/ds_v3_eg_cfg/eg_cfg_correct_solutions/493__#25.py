import math

def calculate_polygons(originX, originY, radius, sides, rings):
    polygons = []
    angle_step = 2 * math.pi / sides
    for ring in range(rings):
        for point in range(sides):
            polygon = []
            for corner in range(sides + 1):
                angle = angle_step * corner
                x_offset = radius * (ring + 1) * math.cos(angle + point * angle_step)
                y_offset = radius * (ring + 1) * math.sin(angle + point * angle_step)
                x = originX + x_offset
                y = originY + y_offset
                point_coord = (x, y)
                polygon.append(point_coord)
            polygons.append(polygon)
    return polygons