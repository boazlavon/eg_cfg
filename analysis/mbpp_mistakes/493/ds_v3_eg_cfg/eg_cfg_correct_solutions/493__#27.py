import math

def calculate_polygons(center_x, center_y, size, grid_width, grid_height):
    polygons = []
    for i in range(grid_width):
        for j in range(grid_height):
            x_offset = center_x + size * 3 * (i - grid_width // 2)
            y_offset = center_y + size * math.sqrt(3) * (j - grid_height // 2)
            if i % 2 != 0:
                y_offset += size * math.sqrt(3) / 2
            hexagon = []
            for k in range(6):
                angle_deg = 60 * k - 30  # Adjusted by -30 degrees for pointy-top hex orientation
                angle_rad = math.radians(angle_deg)
                x = x_offset + size * math.cos(angle_rad)
                y = y_offset + size * math.sin(angle_rad)
                # Round to 12 decimal places to handle floating point precision
                hexagon.append((round(x, 12), round(y, 12)))
            # Close the hexagon by repeating first point
            hexagon.append(hexagon[0])
            polygons.append(hexagon)
    return polygons