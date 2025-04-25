from PIL import Image, ImageDraw, ImageFont
import os # Used for creating a dummy image path
import random # Used for generating random colors in the example

# COCO dataset 80 object categories mapping (integer ID to string name)
# Based on standard COCO detection challenge categories
COCO_LABELS = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
    40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
    44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife',
    50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich',
    55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza',
    60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant',
    65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop',
    74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave',
    79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
    85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
    90: 'toothbrush'
}
# Note: Some IDs are skipped as per the official COCO dataset definition.

def draw_bounding_boxes(image, boxes, scores, labels, label_map=None, colors=None, text_color='white', font_size=15, default_color='red'):
    """
    Draws multiple bounding boxes with scores and text labels on a PIL Image.

    Args:
        image (PIL.Image.Image): The image to draw on.
        boxes (list or array-like): A list or array of bounding boxes.
                                     Expected shape [N, 4], where each element is
                                     (top_left_x, top_left_y, bottom_right_x, bottom_right_y).
        scores (list or array-like): A list or array of confidence scores. Expected shape [N].
        labels (list or array-like): A list or array of integer labels. Expected shape [N].
        label_map (dict, optional): A dictionary mapping integer labels to string names.
                                    If None or a label is not found, the integer label is used.
                                    Defaults to None.
        colors (list or str, optional): A list of colors for each bounding box, or a single
                                        color string to use for all boxes. If None, uses
                                        `default_color` for all. Defaults to None.
        text_color (str): The color of the label/score text. Default is 'white'.
        font_size (int): The font size for the label and score text. Default is 15.
        default_color (str): The default color to use for boxes if `colors` is None
                             or not provided per box. Default is 'red'.


    Returns:
        PIL.Image.Image: The image with bounding boxes and text drawn on it.

    Raises:
        ValueError: If the lengths of boxes, scores, and labels do not match.
    """
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError("Input lists (boxes, scores, labels) must have the same length.")

    if colors and isinstance(colors, list) and len(colors) != len(boxes):
        raise ValueError("If providing a list of colors, it must match the number of boxes.")

    if label_map is None:
        label_map = COCO_LABELS
    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        print("Arial font not found, using PIL default font.")
        try:
            # Try loading a common sans-serif font as a fallback
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            print("Using DejaVuSans font.")
        except IOError:
            print("DejaVuSans font not found, using PIL default font.")
            font = ImageFont.load_default() # Load default PIL font if others fail


    # Iterate through each bounding box, score, and label
    for i, (bbox, score, label_id) in enumerate(zip(boxes, scores, labels)):
        # Determine the color for this box
        if isinstance(colors, list):
            box_color = colors[i]
        elif isinstance(colors, str):
            box_color = colors
        else:
            box_color = default_color # Use default if colors is None or invalid

        # Extract coordinates
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = map(int, bbox) # Ensure coords are int

        # Draw the bounding box rectangle
        draw.rectangle([top_left_x, top_left_y, bottom_right_x, bottom_right_y],
                       outline=box_color, width=3)

        # Get the text label from the map, or use the ID if not found/map not provided
        if label_map and label_id in label_map:
            label_text = label_map[label_id]
        else:
            label_text = str(label_id) # Fallback to integer ID as string

        # Prepare the text to display
        text = f"{label_text}: {score:.2f}" # Use text label

        # Calculate text size and position using textbbox for accuracy
        try:
            # Calculate bounding box of the text itself
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError: # Fallback for older Pillow versions
            # textsize is less accurate but works on older versions
            text_width, text_height = draw.textsize(text, font=font)


        # Position the text background slightly above the bounding box
        text_bg_origin_y = top_left_y - text_height - 4 # Add a small gap (4px)
        # Ensure text background doesn't go off the top edge
        if text_bg_origin_y < 0:
            # Place below top edge inside the box if needed
            text_bg_origin_y = top_left_y + 2

        text_bg_origin_x = top_left_x

        # Draw a filled rectangle behind the text for better visibility
        text_bg_rect = [
            text_bg_origin_x, text_bg_origin_y,
            text_bg_origin_x + text_width + 4, text_bg_origin_y + text_height + 4 # Add padding
        ]
        draw.rectangle(text_bg_rect, fill=box_color)

        # Draw the text itself (slightly offset for padding inside the background)
        draw.text((text_bg_origin_x + 2, text_bg_origin_y + 2), text, fill=text_color, font=font)

    return image

# --- Example Usage ---
if __name__ == "__main__":
    # Create a dummy blank image (e.g., 500x500 pixels, light gray background)
    img_width, img_height = 500, 500
    dummy_image = Image.new('RGB', (img_width, img_height), color = '#E0E0E0') # Light gray

    # Define multiple sample bounding boxes, scores, and COCO labels
    # Format: [(tlx, tly, brx, bry), ...]
    sample_boxes = [
        (50, 70, 200, 220),   # Box 1
        (250, 100, 400, 300), # Box 2
        (10, 250, 150, 450),  # Box 3
        (200, 350, 480, 480), # Box 4
        (300, 20, 450, 150)   # Box 5 (label not in COCO_LABELS)
    ]
    sample_scores = [0.85, 0.92, 0.78, 0.65, 0.99]
    sample_labels = [3, 1, 16, 62, 999] # COCO IDs: car, person, bird, chair, 999 (unknown)

    # Provide a list of colors
    box_colors = ['blue', 'green', 'orange', '#FF00FF', 'cyan'] # Blue, Green, Orange, Magenta, Cyan

    # Draw the bounding boxes using the COCO label map
    image_with_boxes = draw_bounding_boxes(
        dummy_image.copy(),
        sample_boxes,
        sample_scores,
        sample_labels,
        label_map=COCO_LABELS, # Pass the label map here
        colors=box_colors,
        font_size=16
    )


    # Save or show the resulting image
    output_path = "image_with_text_labels.png"
    image_with_boxes.save(output_path)
    print(f"Image saved to {output_path}")

    # --- Example with many boxes and random COCO labels ---
    num_boxes = 20
    many_boxes = []
    many_scores = []
    many_labels = []
    many_colors = []
    coco_ids = list(COCO_LABELS.keys()) # Get valid COCO IDs

    for _ in range(num_boxes):
        x1 = random.randint(0, img_width - 50)
        y1 = random.randint(0, img_height - 50)
        x2 = random.randint(x1 + 30, img_width)
        y2 = random.randint(y1 + 30, img_height)
        many_boxes.append((x1, y1, x2, y2))
        many_scores.append(random.uniform(0.3, 0.99)) # Scores between 0.3 and 0.99
        many_labels.append(random.choice(coco_ids)) # Pick a random valid COCO ID
        # Generate a random color
        many_colors.append(f'#{random.randint(0, 0xFFFFFF):06x}')

    many_boxes_image = Image.new('RGB', (img_width, img_height), color = '#F0F0F0') # Even lighter gray
    image_with_many_boxes = draw_bounding_boxes(
        many_boxes_image,
        many_boxes,
        many_scores,
        many_labels,
        label_map=COCO_LABELS, # Use the COCO map
        colors=many_colors,
        font_size=12
    )
    output_path_many = "image_with_many_text_labels.png"
    image_with_many_boxes.save(output_path_many)
    print(f"Image with many text labels saved to {output_path_many}")

    # To display the image directly (might open in an external viewer)
    # try:
    #     image_with_boxes.show()
    #     image_with_many_boxes.show()
    # except Exception as e:
    #     print(f"Could not display image directly: {e}")
