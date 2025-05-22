import math

def calculate_polygons(center_x, center_y, radius, num_columns, num_rows):
    polygons = []
    hex_height = 2 * radius * math.sin(math.pi / 3)
    hex_width = 2 * radius
    horizontal_spacing = hex_width * 0.75
    vertical_spacing = hex_height
    for row in range(num_rows):
        for col in range(num_columns):
            x = center_x + col * horizontal_spacing
            if row % 2 == 1:
                x += horizontal_spacing / 2
            y = center_y + row * vertical_spacing * 0.5
            hexagon = []
            for i in range(6):
                angle_deg = 60 * i - 30
                angle_rad = math.pi / 180 * angle_deg
                hex_x = x + radius * math.cos(angle_rad)
                hex_y = y + radius * math.sin(angle_rad)
                hexagon.append((hex_x, hex_y))
            hexagon.append(hexagon[0])
            polygons.append(hexagon)
    return polygons