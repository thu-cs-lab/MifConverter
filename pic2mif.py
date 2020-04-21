#!/usr/bin/env python3

import sys
import os

import click
import cv2
import mif
import numpy as np


def error(msg):
    print(f'Error: {msg}')
    exit(1)


@click.command()
@click.argument('input', type=click.Path(exists=True))
@click.argument('output', type=click.Path())
@click.option('-m', '--mode', type=click.Choice(['gray', 'rgb']), default='rgb', help='output file color mode')
@click.option('-f', '--force', is_flag=True, help='allow overriding of output file')
@click.option('-c', '--channel-width', type=click.INT, default=3, help='bit width of each channel per pixel')
@click.option('-w', '--word-width', type=click.INT, default=-1, help='bit width of each word in each file, -1 for one pixel per word')
@click.option('-t', '--threshold', type=click.IntRange(0, 255), default = 127, help='threshold of binarization (0 to 255) (valid when channel width = 1)')
@click.option('-r', '--dump-radix', type=click.Choice(['HEX', 'BIN']), default='HEX', help='radix to use when dumping data')
def process(input, output, mode, force, channel_width, word_width, threshold, dump_radix):

    color = mode == 'rgb'

    pixel_width = channel_width * 3 if color else channel_width

    if word_width == -1:
        word_width = pixel_width
    
    if word_width % pixel_width != 0:
        error('word width must be a multiple of pixel width')

    pixel_per_word = word_width // pixel_width

    # print parameters
    print('======Parameters======')
    print(f'Input file: {input}')
    print(f'Output file: {output}')
    print(f'Mode: {mode}')
    print(f'Channel width: {channel_width} bits')
    print(f'Word width: {word_width} bits ({pixel_per_word} pixels)')
    if channel_width == 1:
        print(f'Binarization threshold: {threshold}')
    

    print('=======Output=========')

    if os.path.exists(output) and not force:
        error('output file existed, use --force to overwrite')

    # read image in expected format
    img = cv2.imread(input, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
    if img is None:
        error('OpenCV could read your image file')
    if color:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB

    # check image dimension & channel
    w, h = img.shape[0], img.shape[1]
    c = img.shape[2] if color else 1
    print(f'Image shape: {w} * {h} with {c} color channel(s)')
    if c == 1 and color:
        error('Grayscale images could not be used in RGB mode')
    if c != 1 and c != 3:
        error('Unsupported channel number {c}')

    # check & calculate word count
    pixel_num = w * h
    if pixel_num % pixel_per_word != 0:
        error('the picture could not be divided into whole words')
    word_count = w * h // pixel_per_word
    mem_size = word_count * word_width
    print(f'Memory size: {mem_size} bits')
    print(f'Depth (word count): {word_count}')

    # channels to process
    if color:
        channels = cv2.split(img) # in R, G, B
    else:
        channels = [img] # in grayscale

    def keep_width(pixel):
        return 0

    # initialize memory in pixels
    mem = np.zeros(shape=(mem_size // pixel_width, pixel_width), dtype=np.uint8)

    # keep only required bits in image
    def process_value(value):
        if channel_width == 1: # binarization, use given threshold
            return 1 if value > threshold else 0
        else:
            return value >> (8 - channel_width) # take the highest channel_width bits
    
    # flatten each channel to 1-D array, then process image to bits
    channels = [c.flatten() for c in channels]
    current_pixel = 0
    for i, pixel in enumerate(zip(*channels)): # i: pixel offset, pixel is either (r, g, b) or (gray,)
        for j, c in enumerate(pixel): # j: channel offset
            processed_value = process_value(c)
            for w in range(channel_width):
                mem[i][j * channel_width + w] = 1 if processed_value & (1 << w) != 0 else 0

    # reshape memory to (address, word)
    mem = mem.reshape((word_count, word_width))
    with open(output, 'w') as f:
        mif.dump(mem, f, data_radix=dump_radix)
    
    print('Dump succeeded!')

if __name__ == '__main__':
    process()
