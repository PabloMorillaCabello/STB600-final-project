from pypylon import pylon
import cv2
import numpy as np
import time

# connect to camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()


converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# ----------------------------
# Setup video writer
# ----------------------------
output_filename = "basler_recording.avi"

# You will know frame size after first frame is received
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
fps = 30   # or any desired fps if unknown

video_writer = None
# ----------------------------

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        image = converter.Convert(grabResult)
        frame = image.GetArray()

        # initialize video writer once we know frame size
        if video_writer is None:
            height, width, _ = frame.shape
            video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

        # write current frame to video
        video_writer.write(frame)

        # show live preview (if GUI available)
        cv2.imshow('basler live feed', frame)
        if cv2.waitKey(1) == 27:  # ESC to exit
            break

    grabResult.Release()

camera.StopGrabbing()

# release video writer
if video_writer is not None:
    video_writer.release()

cv2.destroyAllWindows()
print("Video saved to:", output_filename)
