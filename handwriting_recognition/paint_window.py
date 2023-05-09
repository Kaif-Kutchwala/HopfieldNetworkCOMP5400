import pygame
from main import classify

BLUE = (0 , 0 , 255)
WHITE = (255, 255, 255)
GREY = (240,240,240)
(width, height) = (784,784)
BLOCK_SIZE = 28
colored_pixels = []

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Handwriting Recognition')

pygame.display.flip()

def convert_coordinate(x):
    return int((width/BLOCK_SIZE)*(x)/width) * BLOCK_SIZE

def color_pixel( screen, x, y, erase = False ):
    x = convert_coordinate(x)
    y = convert_coordinate(y)
    if erase:
        if (x,y) in colored_pixels:
            colored_pixels.remove((x,y))
    else:
        if (x,y) not in colored_pixels:
            colored_pixels.append((x,y))

def draw_grid(screen):
    for x in range(0, width, BLOCK_SIZE):
        for y in range(0, height, BLOCK_SIZE):
            rect = pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(screen, GREY, rect, 1)

def get_drawing():
    output = [-1 for i in range((width // BLOCK_SIZE)**2)]
    for x,y in colored_pixels:
        x //= BLOCK_SIZE
        y //= BLOCK_SIZE
        output[y*(width//BLOCK_SIZE) + x] = 1
    return output
        

running = True
isPressed = False
drawing = False
while running:
    screen.fill(WHITE)
    draw_grid(screen)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if pygame.key.name(event.key) in ["c", "C"]:
                colored_pixels.clear()
            if pygame.key.name(event.key) in ["g", "G"]:
                print(get_drawing())
            if pygame.key.name(event.key) in ["h", "H"]:
                input = get_drawing()
                classify(input)

        if event.type == pygame.MOUSEBUTTONDOWN:
            ( x, y ) = pygame.mouse.get_pos() # returns the position of mouse cursor
            isPressed = True
            if event.button == 1:
                drawing = True
                color_pixel( screen, x, y )
            elif event.button == 3:
                drawing = False
                color_pixel( screen, x, y, erase=True )

        if event.type == pygame.MOUSEBUTTONUP:
            isPressed = False

        if event.type == pygame.MOUSEMOTION and isPressed == True:
            ( x, y ) = pygame.mouse.get_pos() # returns the position of mouse cursor
            if drawing:
                color_pixel( screen, x, y )
            else:
                color_pixel( screen, x, y, erase=True )
    
    for x,y in colored_pixels:
        pygame.draw.rect( screen, BLUE, ( x, y, BLOCK_SIZE, BLOCK_SIZE ))


    pygame.display.update()
    

