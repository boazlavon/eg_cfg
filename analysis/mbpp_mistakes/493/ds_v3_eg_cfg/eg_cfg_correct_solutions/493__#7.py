import math

def calculate_polygons(origin_x, origin_y, size_width, size_height, num_columns):
    hexagons = []
    
    # Calculate basic geometric properties
    side_length = size_width / (3 * num_columns + 0.5)
    radius = side_length
    half_side = radius * math.sqrt(3) / 2
    hex_height = 2 * radius
    hex_width = math.sqrt(3) * radius
    horizontal_spacing = hex_width * 0.75
    vertical_spacing = hex_height * 0.75
    start_x = origin_x - num_columns * horizontal_spacing / 2
    
    # Generate hexagon centers in a grid pattern
    grid_centers = []
    for row in range(size_height):
        centers_row = []
        y_offset = row * vertical_spacing
        x_offset = row % 2 * horizontal_spacing / 2
        current_x = start_x - x_offset
        current_y = origin_y + y_offset
        
        # Determine number of columns for this row (offset for odd/even rows)
        columns_for_row = num_columns - row % 2
        
        # Calculate center coordinates for each hexagon in this row
        for col in range(columns_for_row):
            hex_x = current_x + col * horizontal_spacing
            hex_y = current_y
            centers_row.append((hex_x, hex_y))
        grid_centers.append(centers_row)
    
    # Calculate vertices for each hexagon
    angle_increment = math.pi / 3  # 60 degrees in radians
    for row_centers in grid_centers:
        for (center_x, center_y) in row_centers:
            vertices = []
            
            # Generate 7 vertices (6 for hexagon + 1 to close the polygon)
            for i in range(7):
                # Angle starts at 30 degrees (math.pi/6) for pointy-top orientation
                angle = math.pi / 6 + i * angle_increment
                
                # Calculate vertex coordinates
                vert_x = center_x + radius * math.cos(angle)
                vert_y = center_y + radius * math.sin(angle)
                
                # Round to handle floating point precision
                rounded_x = round(vert_x, 13)
                rounded_y = round(vert_y, 13)
                
                # Append vertex to vertices list
                vertices