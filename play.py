from keras.models import load_model
import cv2
import numpy as np
from random import choice

REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "none",
    4: "thumbs_up",
    5: "thumbs_down"
}


def mapper(val):
    return REV_CLASS_MAP[val]


def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"

    if move1 == "rock":
        if move2 == "scissors":
            return "User"
        if move2 == "paper":
            return "Computer"

    if move1 == "paper":
        if move2 == "rock":
            return "User"
        if move2 == "scissors":
            return "Computer"

    if move1 == "scissors":
        if move2 == "paper":
            return "User"
        if move2 == "rock":
            return "Computer"


def main():
    model = load_model("rock-paper-scissors-model.h5")

    cap = cv2.VideoCapture("http://192.168.1.138:4747/video", cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    prev_move = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # rectangle for user to play
        cv2.rectangle(frame, (30, 70), (230, 270), (255, 255, 255), 2)
        # rectangle for computer to play
        cv2.rectangle(frame, (400, 70), (600, 270), (255, 255, 255), 2)

        # extract the region of image within the user rectangle
        roi = frame[50:250, 50:250]
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (227, 227))

        # predict the move made
        pred = model.predict(np.array([img]))
        move_code = np.argmax(pred[0])
        user_move_name = mapper(move_code)

        # predict the winner (human vs computer)
        if prev_move != user_move_name:
            if user_move_name != "none":
                computer_move_name = choice(['rock', 'paper', 'scissors'])

                # Generate computer's move icon
                icon = cv2.imread("images/{}.png".format(computer_move_name))

                # Print dimensions for troubleshooting
                print("Icon dimensions:", icon.shape)
                print("Region dimensions:", frame[70:270, 400:600].shape)

                # Resize icon to match the region dimensions
                icon = cv2.resize(icon, (200, 200))

                # Check if the region dimensions match the icon dimensions
                if frame[70:270, 400:600].shape == icon.shape:
                    frame[70:270, 400:600] = icon
                else:
                    print("Error: Icon dimensions do not match the specified region.")
            else:
                computer_move_name = "none"
        prev_move = user_move_name

        # display the information
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Your Move: " + user_move_name,
                    (50, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Computer's Move: " + computer_move_name,
                    (750, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        # Display dimensions mismatch error if applicable
        cv2.putText(frame, "Winner: None",
                    (400, 600), font, 2, (0, 0, 255), 4, cv2.LINE_AA)

        # Display the frame
        cv2.imshow("Rock Paper Scissors", frame)

        k = cv2.waitKey(10)
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
