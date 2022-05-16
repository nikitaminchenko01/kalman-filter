import pygame
import os
import numpy as np
import time

class PygameImage(object):
    __images_path = "images"

    def __init__(self, image_name):
        self.__images_path = os.path.join(self.__images_path, image_name)
        self.image = pygame.image.load(self.__images_path)


class VisualizationObject(object):
    def __init__(self, starting_x, starting_y, pygame_image = None):
        self.x = starting_x
        self.y = starting_y
        self.pygame_image = pygame_image

    def draw(self, window):
        window.blit(self.pygame_image.image, (self.x, self.y))

    def update(self, new_x = None, new_y = None):
        if new_x is not None:
            self.x = new_x
        if new_y is not None:
            self.y += new_y

    def update_by_offsets(self, x_offset, y_offset):
        self.update(self.x + x_offset, self.y + y_offset)


class VisualizationOutline(VisualizationObject):
    def __init__(self, m_object, pygame_image, x_offset, y_offset):
        super().__init__(m_object.x + x_offset, m_object.y + y_offset, pygame_image)
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.m_object = m_object
        self.pygame_image = pygame_image

    def update(self, new_x = None, new_y=None):
        self.x = self.m_object.x + self.x_offset
        self.y = self.m_object.y + self.y_offset


class Visualizer:
    def __init__(self, window_width, window_height):
        self.visualization_objects = []
        self.window_width = window_width
        self.window_height = window_height
        self.window = pygame.display.set_mode((self.window_width, self.window_height))

    def draw(self):
        for visualization_object in self.visualization_objects:
            visualization_object.draw(self.window)


class KalmanFilterVisualizer(Visualizer):
    __DEF_WINDOW_WIDTH = 950
    __DEF_WINDOW_HEIGHT = 437
    __DEF_IMAGE_WIDTH = 150
    __DEF_CAR_Y_POSITION = 325
    __DEF_REAL_CAR_IMAGE_NAME = 'car_green_xm.png'
    __DEF_KALMAN_CAR_IMAGE_NAME = 'car_yellow_xm.png'
    __DEF_MEASURED_CAR_IMAGE_NAME = 'car_red_xm.png'
    __DEF_OUTLINE_CAR_IMAGE_NAME = 'car_green_transparent_xm.png'
    __DEF_OUTLINE_CAR_OFFSET = -50
    __DEF_ERROR_SCALE_FACTOR = 30
    __DEF_SPEED_SCALE_FACTOR = 1
    __DEF_FONT_NAME = 'comicsans'
    __DEF_FONT_SIZE = 30
    __DEF_FONT_COLOR = (0, 0, 0)

    def __init__(self, measured_errors, kalman_filter_errors, real_speeds, estimated_speeds, clock_tick):
        super().__init__(self.__DEF_WINDOW_WIDTH, self.__DEF_WINDOW_HEIGHT)
        self.measured_errors = measured_errors
        self.kalman_filter_errors = kalman_filter_errors
        self.real_speeds = real_speeds
        self.estimated_speeds = estimated_speeds
        self.scaled_measured_errors = self.measured_errors * self.__DEF_ERROR_SCALE_FACTOR
        self.scaled_kalman_errors = self.kalman_filter_errors * self.__DEF_ERROR_SCALE_FACTOR
        self.scaled_speeds = self.real_speeds * self.__DEF_SPEED_SCALE_FACTOR
        self.clock_tick = clock_tick
        pygame.init()
        self.font = pygame.font.SysFont(self.__DEF_FONT_NAME, self.__DEF_FONT_SIZE)
        self.clock = pygame.time.Clock()
        self.init_objects()

    def init_objects(self):
        self.init_background()
        self.init_cars()
        self.init_texts()
        self.init_dots()

    def init_background(self):
        background_image = PygameImage('bg.png')
        background_image.image.convert()
        second_background_image = PygameImage('bg.png')
        second_background_image.image.convert()
        self.background = VisualizationObject(0, 0, background_image)
        self.second_background = VisualizationObject(second_background_image.image.get_width(), 0,
                                                     second_background_image)
        self.visualization_objects.append(self.background)
        self.visualization_objects.append(self.second_background)

    def init_cars(self):
        self.real_car = VisualizationObject(self.window_width / 2 - self.__DEF_IMAGE_WIDTH / 2, self.__DEF_CAR_Y_POSITION,
                                            PygameImage(self.__DEF_REAL_CAR_IMAGE_NAME))
        self.visualization_objects.append(self.real_car)
        self.outline_car = VisualizationOutline(self.real_car, PygameImage(self.__DEF_OUTLINE_CAR_IMAGE_NAME), 0,
                                                self.__DEF_OUTLINE_CAR_OFFSET)
        self.visualization_objects.append(self.outline_car)

        self.kalman_car = VisualizationObject(self.real_car.x, self.real_car.y,
                                              PygameImage(self.__DEF_KALMAN_CAR_IMAGE_NAME))
        self.visualization_objects.append(self.kalman_car)

        self.measured_car = VisualizationObject(self.real_car.x, self.real_car.y,
                                              PygameImage(self.__DEF_MEASURED_CAR_IMAGE_NAME))
        self.visualization_objects.append(self.measured_car)

    def init_texts(self):
        self.measured_error_text = ""
        self.kalman_error_text = ""
        self.estimated_speed_text = ""
        self.real_position_text = "Real position"
        self.measured_position_text = "Measured position"
        self.kalman_filtered_position_text = "Kalman filtered position"

    def init_dots(self):
        self.real_dot = VisualizationObject(620, 10, PygameImage('green.png'))
        self.kalman_dot = VisualizationObject(620, 50, PygameImage('yellow.png'))
        self.measured_dot = VisualizationObject(620, 90, PygameImage('red.png'))
        self.visualization_objects.append(self.real_dot)
        self.visualization_objects.append(self.kalman_dot)
        self.visualization_objects.append(self.measured_dot)

    def run(self):
        pygame.display.set_caption('Kalman Filter Visualisation')
        for i in range(len(self.kalman_filter_errors) - 1):
            self.update_backgrounds_position(self.scaled_speeds[i + 1])
            self.update_cars_position(self.scaled_measured_errors[i + 1], self.scaled_kalman_errors[i + 1])
            self.update_texts(self.measured_errors[i+1], self.kalman_filter_errors[i+1], self.estimated_speeds[i+1])
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    run = False
            self.clock.tick(self.clock_tick)
            self.redraw_window()
            time.sleep(0.1)

    def update_backgrounds_position(self, offset):
        self.background.x -= offset
        self.second_background.x -= offset
        if self.background.x < self.background.pygame_image.image.get_width() * -1:
            self.background.x = self.background.pygame_image.image.get_width()
        if self.second_background.x < self.second_background.pygame_image.image.get_width() * -1:
            self.second_background.x = self.second_background.pygame_image.image.get_width()

    def update_cars_position(self, measured_error, kalman_error):
        self.measured_car.update(self.real_car.x + measured_error)
        self.kalman_car.update(self.real_car.x + kalman_error)

    def update_texts(self, measured_error, kalman_error, estimated_velocity):
        self.measured_error_text = 'Measured error: ' + str("{0:.2f}".format(measured_error)) + ' m'
        self.kalman_error_text = 'Kalman filter error: ' + str("{0:.2f}".format(kalman_error)) + ' m'
        self.estimated_speed_text = 'Estimated speed: ' + str("{0:.2f}".format(estimated_velocity)) + ' m/s'

    def draw_texts(self):
        measured_text = self.font.render(self.measured_error_text, 1, self.__DEF_FONT_COLOR)
        kalman_text = self.font.render(self.kalman_error_text, 1, self.__DEF_FONT_COLOR)
        velocity_text = self.font.render(self.estimated_speed_text, 1, self.__DEF_FONT_COLOR)
        green_text = self.font.render(self.real_position_text, 1, self.__DEF_FONT_COLOR)
        red_text = self.font.render(self.measured_position_text, 1, self.__DEF_FONT_COLOR)
        yellow_text = self.font.render(self.kalman_filtered_position_text, 1, self.__DEF_FONT_COLOR)
        self.window.blit(measured_text, (10, 10))
        self.window.blit(kalman_text, (10, 50))
        self.window.blit(velocity_text, (10, 100))
        self.window.blit(green_text, (670, 18))
        self.window.blit(yellow_text, (670, 58))
        self.window.blit(red_text, (670, 98))

    def redraw_window(self):
        self.draw()
        self.draw_texts()
        pygame.display.update()