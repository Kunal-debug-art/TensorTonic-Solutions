import math

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    anchors = []
    stride = image_size / feature_size

    for i in range(feature_size):
        cy = (i + 0.5) * stride
        for j in range(feature_size):
            cx = (j + 0.5) * stride
            for s in scales:
                for r in aspect_ratios:
                    w = s * math.sqrt(r)
                    h = s / math.sqrt(r)
                    anchors.append([
                        float(cx - w / 2),
                        float(cy - h / 2),
                        float(cx + w / 2),
                        float(cy + h / 2)
                    ])

    return anchors