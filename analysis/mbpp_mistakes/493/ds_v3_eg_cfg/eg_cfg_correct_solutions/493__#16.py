import math

def calculate_polygons(center_x, center_y, side_length, num_rows, num_cols):
    polygons = []
    sqrt_3 = math.sqrt(3)
    hex_height = side_length * 2
    hex_width = sqrt_3 * side_length
    horizontal_step = hex_width
    vertical_step = hex_height * 0.75

    for row in range(num_rows):
        for col in range(num_cols):
            x_offset = col * horizontal_step
            y_offset = row * vertical_step
            if row % 2 == 1:
                x_offset = x_offset + horizontal_step / 2
            hex_center_x = center_x + x_offset
            hex_center_y = center_y + y_offset
            hexagon = []
            
            for i in range(7):
                angle_deg = 60 * i - 30
                angle_rad = math.radians(angle_deg)
                x_point = hex_center_x + side_length * math.cos(angle_rad)
                y_point = hex_center_y + side_length * math.sin(angle_rad)
                point = (x_point, y_point)
                hexagon.append(point)
            
            polygons.append(hexagon)
    
    return polygons