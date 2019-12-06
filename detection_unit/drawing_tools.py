from PIL import ImageDraw

colour = {}
colour[0] = (255, 255, 255)
colour[1] = (255, 0, 0)
colour[2] = (0, 255, 0)
colour[3] = (0, 0, 255)

def drawrect(drawcontext, xy, outline=None, width=0):
    [(x0,y0),(x1,y1)] = xy
    points = (x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)
    drawcontext.line(points, fill=outline, width=width)

def draw_image_with_boxes(image, boxes, labels, show_image=False):

    for [x0, y0, x1, y1], label in zip(boxes, labels):
        drawimage = ImageDraw.Draw(image)
        drawrect(drawimage, [(x0,y0),(x1,y1)], outline=colour[label], width=10)
    if show_image:
        cv2.imshow(cv2.resize(np.asarray(image),(416,416)))
    return image
