import math

def calculate_polygons(initial_x, initial_y, x_spacing, y_spacing, num_hexagons):
    hexagons = []
    radius = x_spacing / 2.0
    for i in range(num_hexagons):
        row = i // 3
        col = i % 3
        cx = initial_x + col * x_spacing
        cy = initial_y + row * y_spacing * math.sqrt(3) / 2
        if col % 2 == 1:
            cy += y_spacing * math.sqrt(3) / 4
        hexagon = []
        for j in range(6):
            angle = math.pi / 3 * j
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            hexagon.append((x, y))
        hexagon.append(hexagon[0])  # Close the hexagon by repeating the first vertex
        hexagons.append(hexagon)
    return hexagons