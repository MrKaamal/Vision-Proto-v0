"""Vision-inspired assistant prototype with speech and face detection."""

from __future__ import annotations

import datetime

import cv2
import pyttsx3


class VisionAssistant:
    """Simple assistant with voice output and webcam face detection."""

    def __init__(self) -> None:
        self.engine = pyttsx3.init()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def speak(self, text: str) -> None:
        print(f"VISION: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

    def greet(self) -> None:
        hour = datetime.datetime.now().hour
        if hour < 12:
            self.speak("Good morning. Vision online.")
        elif hour < 18:
            self.speak("Good afternoon. Vision online.")
        else:
            self.speak("Good evening. Vision online.")

    def detect_faces(self) -> None:
        self.speak("Activating camera for face detection.")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            self.speak("Camera not found.")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.speak("Camera frame read failed.")
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.putText(
                    frame,
                    f"Faces detected: {len(faces)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("VISION Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.speak("Camera shutdown complete.")


def main() -> None:
    assistant = VisionAssistant()
    assistant.greet()
    assistant.detect_faces()


if __name__ == "__main__":
    main()
