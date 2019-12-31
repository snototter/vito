#!/usr/bin/env python
# coding=utf-8
"""Handy visualization tools."""

import cv2
import numpy as np

from .imvis_cpp import *
from . import colormaps


################################################################################
################################################################################
# Image display

def imshow(img, title="Image", wait_ms=10, flip_channels=False):
    """
    Convenience 1-liner to display image and wait for key input.
    :param img:    Should be provided as BGR, otherwise use flip_channels=True
    :param title:  Window title
    :param wait_ms: cv2.waitKey() input
    :param flip_channels: if you want to display an RGB image poperly, we need 
                          to flip the color channels
    :return: Pressed key or -1, i.e. cv2.waitKey() output
    """
    if flip_channels and len(img.shape) == 3:
        disp = img[:,:,::-1]
    else:
        disp = img
    cv2.imshow(title, disp)
    return cv2.waitKey(wait_ms)



################################################################################
################################################################################
# Coloring, image manipulation

def color_by_id(id, flip_channels=False):
    """Returns a color tuple (rgb) to colorize labels, identities, segments, etc."""
    col = color_by_id__(id)
    if flip_channels:
        return (col[2], col[1], col[0])
    return col


def pseudocolor(values, limits=[0.0,1.0], color_map=colormaps.colormap_parula_rgb):
    if len(values.shape) > 2:
        if values.shape[2] > 1:
            raise ValueError('Input to pseudocoloring must be a single channel data matrix, shaped (H,W) or (H,W,1)!')
        values = values.reshape((values.shape[0], values.shape[1]))
    values = values.astype(np.float64)
    lut = np.asarray(color_map)

    def _is_bool():
        try:
            if values.dtype == np.bool or limits[0].dtype == np.bool:
                return True
            return False
        except:
            return False
    if _is_bool():
        lookup_values = 255*values.astype(np.uint8)
    else:
        interval = (limits[1] - limits[0]) / 255.0
        values[values < limits[0]] = limits[0]
        values[values > limits[1]] = limits[1]
        lookup_values = np.floor((values - limits[0]) / interval).astype(np.int32)
    colorized = lut[lookup_values].astype(np.uint8)
    return colorized


# TODO also need this in Cpp
def overlay(img1, img2, weight1, mask1=None):
    """Overlay two images with alpha blending, s.t.
    out = img1 * weight1 + img2 * (1-weight2).
    Optionally, only overlays those parts of the image which are indicated by
    non-zero mask1 pixels: out = blended if mask1 > 0, else img2
    Output dtype will be the same as img2.dtype.
    Only float32, float64 and uint8 are supported.
    """
    if weight1 < 0.0 or weight1 > 1.0:
        raise ValueError('Weight factor must be in [0,1]')

    # Allow overlaying a grayscale image on top of a color image (and vice versa)
    if len(img1.shape) == 2:
        channels1 = 1
    else:
        channels1 = img1.shape[2]

    if len(img2.shape) == 2:
        channels2 = 1
    else:
        channels2 = img2.shape[2]

    if channels1 == channels2 and not (channels1 == 1 or channels1 == 3):
        raise ValueError('Can only extrapolate single channel image to the others dimension')

    if channels1 == 1 and channels2 > 1:
        img1 = np.repeat(img1[:,:,np.newaxis], channels2, axis=2)
    if channels2 == 1 and channels1 > 1:
        img2 = np.repeat(img2[:,:,np.newaxis], channels1, axis=2)

    num_channels = 1 if len(img1.shape) == 2 else img1.shape[2]

    # Convert to float64, [0,1]
    if img1.dtype == np.uint8:
        scale1 = 255.0
    elif img1.dtype in [np.float32, np.float64]:
        scale1 = 1.0
    else:
        raise ValueError('Datatype {} not yet supported'.format(img1.dtype))
    img1 = img1.astype(np.float64) / scale1

    target_dtype = img2.dtype
    if img2.dtype == np.uint8:
        scale2 = 255.0
    elif img2.dtype in [np.float32, np.float64]:
        scale2 = 1.0
    else:
        raise ValueError('Datatype {} not yet supported'.format(img2.dtype))
    img2 = img2.astype(np.float64) / scale2

    if mask1 is None:
        out = weight1 * img1 + (1. - weight1) * img2
    else:
        if num_channels == 1:
            img1 = np.where(np.repeat(mask1[:,:,np.newaxis], num_channels, axis=2) > 0, img1, img2)
        else:
            img1 = np.where(mask1 > 0, img1, img2)
        out = weight1 * img1 + (1. - weight1) * img2
        # out = img2
        # idx = np.where(mask1 > 0)
        # out[idx] = weight1 * img1[idx] + (1. - weight1) * img2[idx]
    return (scale2 * out).astype(target_dtype)


################################################################################
# Convenience wrappers for C++ code

# See pvt/pvt_visualization/drawing.h/.cpp for valid anchors
__text_anchors = {'northwest' : (1|8), 'north' : (2|8), 'northeast' : (4|8),
    'west' : (1|16), 'center' : (2|16), 'east' : (4|16),
    'southwest' : (1|32), 'south' : (2|32), 'southeast' : (4|32)}

def draw_text_box(image, text, pos, text_anchor='northwest',
        bg_color=(0,0,0), font_color=(-1,-1,-1), font_scale=1.0,
        font_thickness=1, padding=5, fill_opacity=0.7):
    """Draws a text box.

    :param image: numpy.array

    :param text:  str

    :param pos:   Position (x,y), see also comments on 'text_anchor' param.

    :param text_anchor: Defines which corner of the text box is given as the
                        'pos' parameter.

    :param bg_color:    RGB/BGR tuple of the text box' background.
    :param font_color:  If (-1,-1,-1), font will have the complementary color to
                        the bg_color. Otherwise specify as RGB/BGR tuple.

    :param font_scale:      OpenCV's putText() font_scale parameter.

    :param font_thickness:  OpenCV's putText() thickness parameter.

    :param padding:         Padding between text box edges and text, must be >= 0.

    :param fill_opacity:    Opacity in [0,1] of the background box.
    """
    return draw_text_box__(image, text=text, pos=pos,
        textbox_anchor=__text_anchors[text_anchor], bg_color=bg_color,
        font_color=font_color, font_scale=font_scale, font_thickness=font_thickness,
        padding=padding, fill_opacity=fill_opacity)


# Needs to be wrapped because of the text-to-param mapping for the text_anchor
def draw_bboxes2d(image, bounding_boxes, default_box_color=(0,0,255),
        line_width=1, fill_opacity=0.0,
        font_color=(-1,-1,-1), text_anchor='northwest', font_scale=1.0, font_thickness=1,
        text_padding=5, text_box_opacity=0.7, non_overlapping=False):
    """Draws a list of bounding boxes.

    :param image: Image to draw upon.

    :param bounding_boxes: list of bounding_boxes [bbox1, bbox2, ...]; where each
        bbox can be a list or a tuple: ([l,t,w,h], color, caption, dashed).

          * color is a three-element tuple.

          * caption is a string which will be displayed in a text box (set
            text_anchor and the font_XXX params correspondingly).

          * dashed is a boolean, indicating whether the bounding box should be
            drawn dashed (currently, with a fixed dash length of 10 px, see C++
            code in pypvt/src/pypvt_visualization.cpp).

    :param default_box_color: if you don't specify the color per bbox, we
        default to this color.

    :param line_width: Width of the bounding box edges (integer).

    :param fill_opacity: between 0.0 and 1.0 (if > 0, the bbox will be filled
        transparently).

    :param font_color: if (-1,-1,-1), the font will have the complementary box
        color, otherwise this is the fixed font color for all boxes.

    :param text_anchor: where to place the caption of each bounding box,
        use lower case "compass points"/"cardinal directions" without spaces, eg
          'northwest', 'north', 'center', 'east', 'southeast', ...

    :param font_scale: scale parameter for OpenCV's putText().

    :param font_thickness: thickness parameter for OpenCV's putText().

    :param text_padding: padding between border of bbox and caption text.

    :param text_box_opacity: opacity in [0,1] of the text box' background.

    :param non_overlapping: Don't draw overlapping parts (assume that rects[0]
        is closest to the viewer, then rects[1], ...)
    """
    return draw_bboxes2d__(image, bounding_boxes,
        default_box_color=default_box_color, line_width=line_width,
        fill_opacity=fill_opacity, font_color=font_color,
        text_anchor=__text_anchors[text_anchor], font_scale=font_scale,
        font_thickness=font_thickness, text_box_padding=text_padding,
        text_box_opacity=text_box_opacity, non_overlapping=non_overlapping)


# Needs to be wrapped because of the text-to-param mapping for the text_anchor
def draw_rotated_bboxes2d(image, bounding_boxes, default_box_color=(0,0,255),
        line_width=1, fill_opacity=0.0,
        font_color=(-1,-1,-1), text_anchor='northwest', font_scale=1.0, font_thickness=1,
        text_padding=5, text_box_opacity=0.7, non_overlapping=False, dash_length=10):
    """Draws a list of rotated bounding boxes.
    :param image: Image to draw on.

    :param bounding_boxes: list of rotated bounding boxes [bbox1, bbox2, ...]; where each
        bbox can be a 5-element list [cx,cy,w,h,angle] or a tuple:
            ([cx,cy,w,h,angle], color, caption, dashed). Note that (cx,cy) must
            be the center of the box, and angle is to be given in degrees (as we
            use OpenCV's RotatedRect under the hood).

          * color is a three-element tuple, RGB or BGR (set colors_as_rgb
            correspondingly).

          * caption is a string which will be displayed in a text box (set
            text_anchor and font_XXX correspondingly).

          * dashed is a boolean, indicating whether the bounding box should be
            drawn dashed (currently, with a fixed dash length of 10 px, see C++
            code in pypvt/src/pypvt_visualization.cpp).

    :param default_box_color: if you don't specify the color per bbox, we
        default to this color.

    :param line_width: Width of the bounding box edges (integer).

    :param fill_opacity: between 0.0 and 1.0 (if > 0, the bbox will be filled
        transparently).

    :param font_color: if (-1,-1,-1), the font will have the complementary box
        color, otherwise this is the fixed font color for all boxes.

    :param text_anchor: where to place the caption of each bounding box,
        use lower case "compass points"/"cardinal directions" without spaces, eg
          'northwest', 'north', 'center', 'east', 'southeast', ...

    :param font_scale: scale parameter for OpenCV's putText().

    :param font_thickness: thickness parameter for OpenCV's putText().

    :param text_padding: padding between border of bbox and caption text.

    :param text_box_opacity: opacity in [0,1] of the text box' background.

    :param non_overlapping: Don't draw overlapping parts (assume that rects[0]
        is closest to the viewer, then rects[1], ...)

    :param dash_length: if a bounding box should be drawn dashed, this defines
        the length of each dash.
    
    :return: image (i.e. np.array)
    """
    return draw_rotated_bboxes2d__(image, bounding_boxes,
        default_box_color=default_box_color, line_width=line_width,
        fill_opacity=fill_opacity, font_color=font_color,
        text_anchor=__text_anchors[text_anchor], font_scale=font_scale,
        font_thickness=font_thickness, text_box_padding=text_padding,
        text_box_opacity=text_box_opacity, non_overlapping=non_overlapping,
        dash_length=dash_length)


# Needs to be wrapped because of the text-to-param mapping for the text_anchor
def draw_bboxes3d(image, bounding_boxes, K, R, t, scale_image_points=1.0,
        default_box_color=(255,0,0), line_width=1, dash_length=10, fill_opacity_top=0.3, fill_opacity_bottom=0.5,
        font_color=(-1,-1,-1), text_anchor='northwest', font_scale=1.0,
        font_thickness=1, text_box_padding=5, text_box_opacity=0.7,
        non_overlapping=False):
        """Renders a list of 3D bounding boxes.
        :param image:
        :param bounding_boxes: list of 3D bounding boxes [bbox1, bbox2, ...];
            where each bbox is a tuple: (coords, color, caption)

              * coords is a list of the 8 bounding box corners in 3D, e.g.
                [(0,0,0), (1,17,-0.5), ...]; the first 4 coordinates belong to
                the top plane, the second 4 to the bottom plane (if top is really
                on top doesn't actually matter...)
                Ensure that the coordinates are provided clockwise, e.g. we
                build the top edge as coords[0]-->coords[1]; its (clockwise)
                neighbor as coords[1]-->coords[2], etc.

              * color is a three-element RGB/BGR tuple.

              * caption is a string which will be displayed in a text box (set
                text_anchor and font_XXX correspondingly).

        :params K, R, t: intrinsics 3x3, rotation 3x3 and translation 3x1 np.array

        :param scale_image_points: if you resized images for visualization, you
            can provide the scaling factor so the projected points will be scaled
            properly.

        :param default_box_color: if you don't specify the color per bbox, we
            default to this color.

        :param line_width: Width of the bounding box edges (integer).

        :param dash_length: Occluded edges will be dashed (unless you provide a
            negative value here). TODO not yet tested

        :param fill_opacity_top: between 0.0 and 1.0 (if > 0, the top plane will
            be filled transparently).

        :param fill_opacity_bottom: same for the bottom...

        :param font_color: if (-1,-1,-1), the font will have the complementary box
            color, otherwise this is the fixed font color for all boxes.

        :param text_anchor: where to place the caption of each bounding box,
            use lower case "compass points"/"cardinal directions" without spaces, eg
              'northwest', 'north', 'center', 'east', 'southeast', ...

        :param font_scale: scale parameter for OpenCV's putText().

        :param font_thickness: thickness parameter for OpenCV's putText().

        :param text_padding: padding between border of bbox and caption text.

        :param text_box_opacity: opacity in [0,1] of the text box' background.

        :param non_overlapping: Don't draw overlapping parts (assume that rects[0]
            is closest to the viewer, then rects[1], ...)
        """
        return draw_bboxes3d__(image, bounding_boxes, K, R, t,
            scale_image_points=scale_image_points, default_box_color=default_box_color,
            line_width=line_width, dash_length=dash_length, 
            fill_opacity_top=fill_opacity_top, fill_opacity_bottom=fill_opacity_bottom,
            font_color=font_color, text_anchor=__text_anchors[text_anchor],
            font_scale=font_scale, font_thickness=font_thickness, text_box_padding=text_box_padding,
            text_box_opacity=text_box_opacity, non_overlapping=non_overlapping)



def draw_points(image, points, color=(-1.0,-1.0,-1.0), radius=5, line_width=-1, opacity=0.8):
    """Draw the 2xN points given as numpy array or list/tuple of (x,y) tuples onto the image.

    :param image: input image.

    :param points: 2xN numpy.array or list of tuples: [(x0,y0), (x1,y1), ...].

    :param color: may be a single RGB/BGRE tuple or a list of RGB/BGR tuples.
                  If it's a single tuple and (-1,-1,-1), the points will be drawn
                  by alternating colors.
                  If it's a list of tuples, points[i] will be drawn with color[i].
                  Otherwise (color is a single tuple), all points are drawn with 
                  the same color.
    
    :param radius:     radius of drawn circles.

    :param line_width: negative for filled circle, otherwise line thickness.

    :param opacity:    opacity in [0,1]

    :return: image (numpy.array)
    """
    if type(points) is np.ndarray:
        pts = [(points[0,i], points[1,i]) for i in range(points.shape[1])]
    elif type(points) is list:
        pts = points
    else:
        raise ValueError('draw_points expects a list or numpy.ndarray as input')
    return draw_points__(image, pts, color, radius, line_width, opacity)

