import pygame
from main import classify

# Define colors
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
GREY = (240, 240, 240)

# Define screen height and width
(width, height) = (784, 784)

# Define size of each cell in grid
BLOCK_SIZE = 28

# Create a list of pixels that are colored
colored_pixels = []

# Initialise pygame display to screen dimensions
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Handwriting Recognition')

pygame.display.flip()


# Converts a coordinate on the screen to a cell coordinate on the grid
def convert_coordinate(x):
    return int((width/BLOCK_SIZE)*(x)/width) * BLOCK_SIZE


# Adds cell to colored_pixels
def color_cell(x, y, erase=False):
    # Get cell coordinates from screen coordinates
    x = convert_coordinate(x)
    y = convert_coordinate(y)

    # If action is to erase
    if erase:
        # check if cell is in colord_pixels and remove if it is
        if (x, y) in colored_pixels:
            colored_pixels.remove((x, y))

    # Else action is to color
    else:
        # check if cell is in colored_pixels and add if not
        if (x, y) not in colored_pixels:
            colored_pixels.append((x, y))


# Draws a grid on the screen with each cell sized BLOCK_SIZE X BLOCK_SIZE pixels
def draw_grid(screen):
    for x in range(0, width, BLOCK_SIZE):
        for y in range(0, height, BLOCK_SIZE):
            rect = pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(screen, GREY, rect, 1)


# Returns the current drawing as a 784x1 array of 1s and -1s for the hopfield network
def get_drawing_as_hopfield_input():
    # initialise array with -1s
    output = [-1 for i in range((width // BLOCK_SIZE)**2)]
    # for every colored_pixel
    for x, y in colored_pixels:
        # find x,y coordinates in range 0-28
        x //= BLOCK_SIZE
        y //= BLOCK_SIZE
        # set element in output to 1
        output[y*(width//BLOCK_SIZE) + x] = 1
    return output


# Flag to set status of the window
running = True

# Flag for checking if any mouse button is pressed
isPressed = False

# Flag for checking if user is drawing or erasing
drawing = False

# Game loop
while running:
    # Set background to WHITE
    screen.fill(WHITE)
    # draw the grid
    draw_grid(screen)

    # handle events
    for event in pygame.event.get():
        # if user quits, end
        if event.type == pygame.QUIT:
            running = False

        # if user presses a key
        if event.type == pygame.KEYDOWN:
            # if 'c' is pressed clear the grid
            if pygame.key.name(event.key) in ["c", "C"]:
                colored_pixels.clear()

            # if 'g' is pressed print the drawing as HN input pattern
            if pygame.key.name(event.key) in ["g", "G"]:
                print(get_drawing_as_hopfield_input())

            # if 'h' is pressed classify current drawing using HN
            if pygame.key.name(event.key) in ["h", "H"]:
                input = get_drawing_as_hopfield_input()
                classify(input)

        # if user presses a mouse button
        if event.type == pygame.MOUSEBUTTONDOWN:
            # get position of mouse cursor
            (x, y) = pygame.mouse.get_pos()
            # set isPressed flag
            isPressed = True

            # if left mouse button is pressed
            if event.button == 1:
                # user is drawing
                drawing = True
                # color the cell cursor is in
                color_cell(x, y)

            # if right mouse button is pressed
            elif event.button == 3:
                # user is erasing
                drawing = False
                # clear the cell cursor is in
                color_cell(x, y, erase=True)

        # if user releases mouse button
        if event.type == pygame.MOUSEBUTTONUP:
            # reset isPressed flag
            isPressed = False

        # if user moves cursor while mouse button is pressed
        if event.type == pygame.MOUSEMOTION and isPressed == True:
            # get position of mouse cursor
            (x, y) = pygame.mouse.get_pos()

            # fill cell if user is drawing else clear
            if drawing:
                color_cell(x, y)
            else:
                color_cell(x, y, erase=True)

    # fill in all colored pixels
    for x, y in colored_pixels:
        pygame.draw.rect(screen, BLUE, (x, y, BLOCK_SIZE, BLOCK_SIZE))

    # update the display
    pygame.display.update()
