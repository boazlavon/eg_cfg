import math

def calculate_polygons(startx, starty, endx, endy, radius):
    polygons = []
    sqrt3 = math.sqrt(3)
    diameter = 2 * radius
    offset_x = diameter * 3 / 4
    offset_y = diameter * sqrt3 / 2
    
    current_x = startx
    current_y = starty
    
    while current_y <= endy:
        while current_x <= endx:
            hexagon = []
            for i in range(6):
                angle_deg = 60 * i - 30
                angle_rad = math.pi / 180 * angle_deg
                x = current_x + radius * math.cos(angle_rad)
                y = current_y + radius * math.sin(angle_rad)
                hexagon.append((x, y))
            hexagon.append(hexagon[0])
            polygons.append(hexagon)
            current_x = current_x + offset_x
        current_x = startx
        current_y = current_y + offset_y
    
    return polygons