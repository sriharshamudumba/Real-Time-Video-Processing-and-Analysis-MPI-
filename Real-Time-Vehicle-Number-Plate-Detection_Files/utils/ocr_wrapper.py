def detect_text(frame, reader):
    results = reader.readtext(frame)
    output = []
    for (bbox, text, _) in results:
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        output.append(((top_left, bottom_right), text))
    return output
