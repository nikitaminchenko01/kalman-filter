import pygame
from pygame.locals import *
import os
import sys
import math
import random
import numpy as np


def run(kalman_errors, measured_errors, velocity, kalman_velocity):
    W, H = 1200, 437
    imW = 150
    win = pygame.display.set_mode((W, H))
    pygame.display.set_caption('Kalman Filter Visualisation')

    bg = pygame.image.load(os.path.join('images', 'bg.png')).convert()
    bgX = 0
    bgX2 = bg.get_width()

    class player(object):

        def __init__(self, x, y, image):
            self.x = x
            self.y = y
            self.image = image

        def draw(self, win):
            win.blit(self.image, (self.x, self.y))

    pygame.init()

    image_real = pygame.image.load(os.path.join('images', 'car_green_xm.png'))
    image_kalman = pygame.image.load(os.path.join('images', 'car_yellow_xm.png'))
    image_measured = pygame.image.load(os.path.join('images', 'car_red_xm.png'))
    image_green_outline = pygame.image.load(os.path.join('images', 'car_green_transparent_xm.png'))


    image_green = pygame.image.load(os.path.join('images', 'green.png'))
    image_red = pygame.image.load(os.path.join('images', 'red.png'))
    image_yellow = pygame.image.load(os.path.join('images', 'yellow.png'))



    clock = pygame.time.Clock()
    run = True
    real = player(W / 2 - imW / 2, 325, image_real)
    measured = player(real.x + measured_errors[0], real.y, image_measured)
    kalman = player(real.x + kalman_errors[0], real.y, image_kalman)
    outline = player(real.x, real.y - 50, image_green_outline)



    def redrawWindow(kalman_error, measured_error, estimated_velocity):
        largeFont = pygame.font.SysFont('comicsans', 30)
        win.blit(bg, (bgX, 0))
        win.blit(bg, (bgX2, 0))
        real.draw(win)
        measured.draw(win)
        kalman.draw(win)
        outline.draw(win)
        measured_text = largeFont.render('Measured error: ' + str("{0:.2f}".format(measured_error)) + ' m', 1, (0, 0, 0))
        kalman_text = largeFont.render('Kalman filter error: ' + str("{0:.2f}".format(kalman_error)) + ' m', 1, (0, 0, 0))
        velocity_text = largeFont.render('Estimated velocity: ' + str("{0:.2f}".format(estimated_velocity)) + ' m/s', 1, (0, 0, 0))
        green_text = largeFont.render('Real position', 1, (0, 0, 0))
        red_text = largeFont.render('Measured position', 1, (0, 0, 0))
        yellow_text = largeFont.render('Kalman filtered position', 1, (0, 0, 0))
        win.blit(measured_text, (0, 10))
        win.blit(kalman_text, (0, 50))
        win.blit(velocity_text, (0, 100))

        win.blit(image_green, (900, 10))
        win.blit(image_yellow, (900, 50))
        win.blit(image_red, (900, 90))
        win.blit(green_text, (945, 18))
        win.blit(yellow_text, (945, 58))
        win.blit(red_text, (945, 98))
        pygame.display.update()


    used_kalman_errors = kalman_errors * 30
    used_measured_errors = measured_errors * 30
    used_velocities = velocity
    for i in range(len(kalman_errors) - 1):
        bgX -= used_velocities[i + 1]
        bgX2 -= used_velocities[i + 1]
        measured.x = real.x + used_measured_errors[i + 1]
        kalman.x = real.x + used_kalman_errors[i + 1]

        if bgX < bg.get_width() * -1:
            bgX = bg.get_width()
        if bgX2 < bg.get_width() * -1:
            bgX2 = bg.get_width()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                run = False
        clock.tick(100)
        # if i > 50:
        #     redrawWindow(kalman_errors[i + 1], np.average(np.absolute(measured_errors[i-49:i+1])),  velocity[i + 1])
        # else:
        redrawWindow(kalman_errors[i + 1], measured_errors[i + 1], velocity[i + 1])