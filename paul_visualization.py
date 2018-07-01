# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

NUM_COLORS = len(STANDARD_COLORS)

try:
  FONT = ImageFont.truetype('/usr/share/fonts/truetype/ubuntu-font-family/UbuntuMono-R.ttf', 24)
  # FONT = ImageFont.truetype('arial.ttf', 24)
except IOError:
  FONT = ImageFont.load_default()

def _draw_single_box(image, xmin, ymin, xmax, ymax, display_str='', font=FONT, color='black', thickness=4):
  draw = ImageDraw.Draw(image)
  (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
  if display_str != '':
      text_bottom = bottom
      # Reverse list and print from bottom to top.
      text_width, text_height = font.getsize(display_str)
      margin = np.ceil(0.05 * text_height)
      draw.rectangle(
          [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                            text_bottom)],
          fill=color)
      draw.text(
          (left + margin, text_bottom - text_height - margin),
          display_str,
          fill='black',
          font=font)

  return image

def draw_bounding_boxes(image, gt_boxes, im_info, class_names=None):
  """
  To use this function not only for GT visualization but for detection visualization,
  gt_boxes can be N x 5 (XYXYC) or N x 6 (XYXYCS), where C is class and S is score.
  (paulkwon)
  """
  if class_names is not None: class_names = [cl.decode('ascii') for cl in class_names]
  gt_boxes_new = gt_boxes.copy()
  # gt_boxes_new[:, :4] = np.round(gt_boxes_new[:, :4] / im_info[2])
  image = image.copy()
  disp_image = Image.fromarray(np.uint8(image[0]))

  if gt_boxes_new.shape[1] > 4:
    if class_names is not None:   names = [class_names[int(cl)] for cl in gt_boxes_new[:, 4]]
    else:                         names = ['C%02d' % int(cl) for cl in gt_boxes_new[:, 4]]
  else:                           names = [''] * gt_boxes_new.shape[0]

  if gt_boxes_new.shape[1] > 5:
    scores = ['%.2f' % sc for sc in gt_boxes_new[:, 5]]
  else:
    scores = [None] * gt_boxes_new.shape[0]

  for i in range(gt_boxes_new.shape[0]):
    this_class = int(gt_boxes_new[i, 4])
    # text = '%s-%s' % (names[i], scores[i])
    text = '-'.join(filter(None, [names[i], scores[i]]))
    # import pdb; pdb.set_trace()
    disp_image = _draw_single_box(disp_image,
                                gt_boxes_new[i, 0],
                                gt_boxes_new[i, 1],
                                gt_boxes_new[i, 2],
                                gt_boxes_new[i, 3],
                                text, #'N%02d-' % i + class_name,
                                FONT,
                                color=STANDARD_COLORS[this_class % NUM_COLORS])

  image[0, :] = np.array(disp_image)
  return image


def draw_roi_boxes(image, roi_boxes, roi_scores, roi_labels, im_info,
                   class_names=None, thresh=0.):

  assert roi_boxes.shape[0] == roi_scores.shape[0] == roi_labels.shape[0]
  assert roi_boxes.shape[1] == 4

  if class_names is not None: class_names = [cl.decode('ascii') for cl in class_names]

  roi_scores_new = roi_scores.copy()
  gt_boxes_new = roi_boxes.copy()
  roi_labels_new = roi_labels.copy()
  # gt_boxes_new = np.round(gt_boxes_new / im_info[2])
  # gt_boxes_new[:, 1:] = np.round(gt_boxes_new[:, 1:].copy() / im_info[2])
  image = image.copy()
  disp_image = Image.fromarray(np.uint8(image[0]))

  # the first column indicates the image ids in a batch.
  # idxs = gt_boxes_new[:, 0] == 0
  # gt_boxes_new = gt_boxes_new[idxs, 1:]
  # roi_scores_new = roi_scores_new[idxs]
  # roi_labels_new = roi_labels_new[idxs]

  if class_names is not None:   names = [class_names[int(cl)] for cl in roi_labels]
  else:                         names = [None] * roi_scores_new.shape[0]

  for i in range(len(roi_scores_new)):
    if roi_scores_new[i] < thresh: continue
    this_class = int(roi_labels_new[i])
    thickness = 1 if this_class == 0 else 4
    disp_image = _draw_single_box(disp_image,
                                  gt_boxes_new[i, 0],
                                  gt_boxes_new[i, 1],
                                  gt_boxes_new[i, 2],
                                  gt_boxes_new[i, 3],
                                  '-'.join(filter(None, [names[i], '%.2f' % roi_scores[i]])),  #,'%.2f' % roi_scores_new[i]
                                  FONT,
                                  color=STANDARD_COLORS[this_class % NUM_COLORS],
                                  thickness=thickness)

  image[0, :] = np.array(disp_image)
  return image


def draw_bounding_boxes_with_pose(image, gt_boxes, im_info, class_names=None, gt_markers=None):
  """
  To use this function not only for GT visualization but for detection visualization,
  gt_boxes can be N x 5 (XYXYC) or N x 6 (XYXYCS), where C is class and S is score.
  (paulkwon)
  """
  if class_names is not None: class_names = [cl.decode('ascii') if isinstance(cl, bytes) else cl for cl in class_names ]
  gt_boxes_new = gt_boxes.copy()
  # gt_boxes_new[:, :4] = np.round(gt_boxes_new[:, :4] / im_info[2])
  image = image.copy()
  disp_image = Image.fromarray(np.uint8(image[0]))

  if gt_boxes_new.shape[1] > 4:
    if class_names is not None:   names = [class_names[int(cl)] for cl in gt_boxes_new[:, 4]]
    else:                         names = [''] * gt_boxes_new.shape[0] #names = ['C%02d' % int(cl) for cl in gt_boxes_new[:, 4]]
  else:                           names = [''] * gt_boxes_new.shape[0]

  if gt_boxes_new.shape[1] > 5:
    scores = ['%.2f' % sc for sc in gt_boxes_new[:, 5]]
  else:
    scores = [None] * gt_boxes_new.shape[0]

  for i in range(gt_boxes_new.shape[0]):
    this_class = int(gt_boxes_new[i, 4])
    text = '-'.join(filter(None, [names[i], scores[i]]))
    disp_image = _draw_single_box(disp_image,
                                gt_boxes_new[i, 0],
                                gt_boxes_new[i, 1],
                                gt_boxes_new[i, 2],
                                gt_boxes_new[i, 3],
                                text, #'N%02d-' % i + class_name,
                                FONT,
                                color=STANDARD_COLORS[this_class % NUM_COLORS],
                                thickness=1)

    draw = ImageDraw.Draw(disp_image)
    x = (1 - gt_markers[i]) * gt_boxes_new[i, 0] + gt_markers[i] * gt_boxes_new[i, 2]
    y = 0.2 * gt_boxes_new[i, 1] + 0.8 * gt_boxes_new[i, 3]
    draw.line([(x, y), (x, gt_boxes_new[i, 3])],
              width=3,
              fill=STANDARD_COLORS[this_class % NUM_COLORS])

  image[0, :] = np.array(disp_image)
  return image



FACE_COLORS = [
    'FireBrick', 'Salmon', 'LightGoldenRodYellow', 'Green',
    'SteelBlue', 'Navy', 'MediumPurple', 'DeepPink']

def draw_bounding_boxes_with_face(image, gt_boxes, gt_faces):
  """
  To use this function not only for GT visualization but for detection visualization,
  gt_boxes can be N x 5 (XYXYC) or N x 6 (XYXYCS), where C is class and S is score.
  (paulkwon)
  """
  gt_boxes_new = gt_boxes.copy()
  gt_boxes_new = np.round(gt_boxes_new)
  image = image.copy()
  disp_image = Image.fromarray(np.uint8(image[0]))

  names = [None] * gt_boxes_new.shape[0]
  scores = [None] * gt_boxes_new.shape[0]

  for i in range(gt_boxes_new.shape[0]):
    face = gt_faces[i]
    text = '-'.join(filter(None, [names[i], scores[i]]))
    disp_image = _draw_single_box(disp_image,
                                  gt_boxes_new[i, 0],
                                  gt_boxes_new[i, 1],
                                  gt_boxes_new[i, 2],
                                  gt_boxes_new[i, 3],
                                  text,
                                  FONT,
                                  color=FACE_COLORS[face],
                                  thickness=3)

  image[0, :] = np.array(disp_image)
  return image


def draw_bounding_boxes_with_marker_and_angle(image, gt_boxes, gt_markers, gt_angles):
  gt_boxes_new = gt_boxes.copy()
  gt_boxes_new = np.round(gt_boxes_new)
  image = image.copy()
  disp_image = Image.fromarray(np.uint8(image[0]))

  for i in range(gt_boxes_new.shape[0]):
    disp_image = _draw_single_box(disp_image,
                                  gt_boxes_new[i, 0],
                                  gt_boxes_new[i, 1],
                                  gt_boxes_new[i, 2],
                                  gt_boxes_new[i, 3],
                                  '',
                                  FONT,
                                  color=STANDARD_COLORS[1],
                                  thickness=3)

    x = (1 - gt_markers[i]) * gt_boxes_new[i, 0] + gt_markers[i] * gt_boxes_new[i, 2]
    y = gt_boxes_new[i, 3]
    r = 3
    draw = ImageDraw.Draw(disp_image)
    draw.ellipse((x-r, y-r, x+r, y+r), fill=(255, 0, 0, 255))

    # angle visualization length based on box height and width
    h = (gt_boxes_new[i, 3] - gt_boxes_new[i, 1]) * 0.3
    w = gt_boxes_new[i, 2] - gt_boxes_new[i, 0]
    l = h / np.sin(gt_angles[i])
    l = np.abs(l)
    l = np.minimum(l, w)

    dx = l * np.cos(gt_angles[i])
    dy = l * np.sin(gt_angles[i])
    draw.line([(x, y), (x + dx, y + dy)],
              width=2,
              fill=(255, 0, 0, 255))

  image[0, :] = np.array(disp_image)
  return image




from shapely.geometry import LineString, Polygon



def draw_bounding_boxes_with_face_and_angle(image, gt_boxes, gt_markers, gt_angles):
  """
  gt_markers:
  gt_angles:
  """
  gt_boxes_new = gt_boxes.copy()
  gt_boxes_new = np.round(gt_boxes_new)
  image = image.copy()
  disp_image = Image.fromarray(np.uint8(image[0]))

  for i in range(gt_boxes_new.shape[0]):
    disp_image = _draw_single_box(disp_image,
                                  gt_boxes_new[i, 0],
                                  gt_boxes_new[i, 1],
                                  gt_boxes_new[i, 2],
                                  gt_boxes_new[i, 3],
                                  '',
                                  FONT,
                                  color=STANDARD_COLORS[1],
                                  thickness=3)

    marker = gt_markers[i]
    marker = np.clip(marker, 0.02, 0.98)
    if marker < 0.45:
      marker_start = 0.
      marker_end = marker * 2
      p = marker_end
    elif marker > 0.55:
      marker_start = 2 * marker - 1
      marker_end = 1.
      p = marker_start
    else:
      marker_start = 0.
      marker_end = 1.
      p = None

    def get_value_from_ratio(v, ends):
      return (1. - v) * ends[0] + v * ends[1]

    x1 = get_value_from_ratio(marker_start, gt_boxes_new[i, [0, 2]])
    x2 = get_value_from_ratio(marker_end, gt_boxes_new[i, [0, 2]])
    y1 = gt_boxes_new[i, 1]
    y2 = gt_boxes_new[i, 3]

    disp_image = _draw_polygon(disp_image,
                               [[x1, y1], [x1, y2], [x2, y2], [x2, y1]],
                               'FireBrick', alpha=125)
    if p is not None:
      xp = get_value_from_ratio(marker, gt_boxes_new[i, [0, 2]])
      l = 500  # TEMP. should be enough long
      p1 = np.array([xp - l * np.cos(gt_angles[i]), y2 - l * np.sin(gt_angles[i])])
      p2 = np.array([xp + l * np.cos(gt_angles[i]), y2 + l * np.sin(gt_angles[i])])
      line = LineString([tuple(p1), tuple(p2)])
      box = Polygon([[gt_boxes_new[i, 0], gt_boxes_new[i, 1]],
                     [gt_boxes_new[i, 0], gt_boxes_new[i, 3]],
                     [gt_boxes_new[i, 2], gt_boxes_new[i, 3]],
                     [gt_boxes_new[i, 2], gt_boxes_new[i, 1]],
                     [gt_boxes_new[i, 0], gt_boxes_new[i, 1]]])
      intercepts = line.intersection(box)
      draw = ImageDraw.Draw(disp_image)

      try:
        assert(len(intercepts.coords) == 2)
      except:
        import pdb; pdb.set_trace()

      draw.line([tuple(intercepts.coords[0]), tuple(intercepts.coords[1])],
                width=3,
                fill='FireBrick')

  image[0, :] = np.array(disp_image)
  return image





"""
Primitive drawing functions
"""

def _draw_box(image, box, display_str='', font=FONT, color='black', thickness=4, bgalpha=125):
  draw = ImageDraw.Draw(image)
  left, bottom, right, top = box
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
  if display_str != '':
      text_bottom = bottom
      # Reverse list and print from bottom to top.
      text_width, text_height = font.getsize(display_str)
      margin = np.ceil(0.05 * text_height)
      coords = [(left, text_bottom - text_height - 2 * margin),
                (left, text_bottom),
                (left + text_width, text_bottom),
                (left + text_width, text_bottom - text_height - 2 * margin)]
      image = _draw_polygon(image, coords, color, alpha=bgalpha)
      draw = ImageDraw.Draw(image)
      draw.text(
          (left + margin, text_bottom - text_height - margin),
          display_str,
          fill='black',
          font=font)

  return image


def _draw_polygon_bd(im, coords, color, alpha=125):
    draw = ImageDraw.Draw(im)
    coords = [tuple(c) for c in coords]
    draw.polygon(coords, fill=False)
    return im


def _draw_polygon(im, coords, color, alpha=125):
    coords = [tuple(c) for c in coords]
    color_layer = Image.new('RGB', im.size, color)
    alpha_mask = Image.new('L', im.size, 0)
    alpha_mask_draw = ImageDraw.Draw(alpha_mask)
    alpha_mask_draw.polygon(coords, fill=alpha)
    return Image.composite(color_layer, im, alpha_mask)


def _draw_text(im, x, y, display_str, font=FONT, fgcolor='white', color='black',
               thickness=4, alpha=125, align='center'):

    # Reverse list and print from bottom to top.
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)

    if align == 'center':
        text_left = x - text_width * 0.5
        text_top = y - text_height * 0.5
        text_right = x + text_width * 0.5
        text_bottom = y + text_height * 0.5
    elif align == 'left':
        text_left = x
        text_top = y - text_height * 0.5
        text_right = x + text_width
        text_bottom = y + text_height * 0.5

    bg_box = [(text_left, text_top),
              (text_right, text_top),
              (text_right, text_bottom),
              (text_left, text_bottom)]
    im = _draw_polygon(im, bg_box, color, alpha)

    draw = ImageDraw.Draw(im)
    draw.text(
        (text_left, text_top),
        display_str,
        fill=fgcolor,
        font=font)
    return im