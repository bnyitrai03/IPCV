import cv2

cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

try:
    if not cap.isOpened():
        raise RuntimeError("Could not open /dev/video0")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Print the settings OpenCV actually reports.
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    pixel_format = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))

    print("Format:", pixel_format)
    print(
        "Resolution:",
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "x",
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    print("Reported FPS:", cap.get(cv2.CAP_PROP_FPS))

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to receive a frame.")
            break

        cv2.imshow("Webcam - press Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()