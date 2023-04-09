import pygame
import numpy as np
from tensorflow.keras.datasets import mnist
import json

def get_average_matrix(matrices):
    average_matrix = []
    no_of_matrices = len(matrices)
    for i in range(len(matrices[0])):
        sum = 0
        for j in range(no_of_matrices):
            sum += matrices[j][i]
        average_matrix.append(sum/no_of_matrices)
    # print(average_matrix)
    return average_matrix

def separate_digits(images, labels):
    data = {}
    for i in range(1,10):
        data[str(i)] = []
    for i in range(len(labels)):
        if str(labels[i]) not in data.keys():
            continue
        else:
            im = np.reshape(images[i], (1, 784)).tolist()
            data[str(labels[i])].append(im[0])
    return data

def calculate_average_digit_matrices():
    (images, labels), _ = mnist.load_data()
    digits = separate_digits(images, labels)
    averages = {}

    for i in range(1,10):
        averages[str(i)] = get_average_matrix(digits[str(i)])

    file = open("mnist_digits_average.json", "w+")
    file.write(json.dumps(averages))
    file.close()

def display_average_matrix_for(digit):
    file = open("mnist_digits_average.json", "r")
    data = json.loads(file.read())
    file.close()
    average_matrix = data[str(digit)]
    print_matrix(average_matrix, 28, 20, 1)

def print_matrix(matrix, break_point, BLOCK_SIZE, max, no_color = False):
    width = BLOCK_SIZE * break_point
    height = (len(matrix)//break_point) * BLOCK_SIZE
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Average Matrix')
    draw_grid(screen, BLOCK_SIZE, width, height)


    pygame.display.flip()
    running = True
    while running:
        screen.fill((255,255,255))
        draw_grid(screen, BLOCK_SIZE, width, height)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        for i in range(0, len(matrix)):
            (x,y) = get_coordinates(i, height,width, BLOCK_SIZE)
            if no_color:
                color = convert_range(matrix[i], 0, max, 255, 0)
            else:
                color = matrix[i]
            color_pixel(screen, x, y, (color, color, color), BLOCK_SIZE)
        pygame.display.update()
             
def draw_grid(screen, BLOCK_SIZE, width, height):
    for x in range(0, width, BLOCK_SIZE):
        for y in range(0, height, BLOCK_SIZE):
            rect = pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(screen, (240,240,240), rect, 1)

def get_coordinates(index,height, width, BLOCK_SIZE):
    x = index * BLOCK_SIZE % width
    y = (index * BLOCK_SIZE) // width * BLOCK_SIZE
    return (x, y)

def color_pixel( screen, x, y, color, BLOCK_SIZE ):
    # print(color)
    pygame.draw.rect( screen, color, ( x, y, BLOCK_SIZE, BLOCK_SIZE ))

def convert_range(OldValue, OldMin, OldMax, NewMin, NewMax):
    return (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin


