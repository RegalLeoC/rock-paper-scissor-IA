from keras.models import load_model
import cv2
import numpy as np
from random import choice
import time

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
    score = {'wins': 0, 'losses': 0, 'ties': 0}

    # Initialize timer variables
    start_time = time.time()
    wait_time = 0  # 0 seconds

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

        # Check if the timer has expired
        if time.time() - start_time >= wait_time:
            # Reset the timer and process user's move
            start_time = time.time()

            # predict the winner (human vs computer)
            if prev_move != user_move_name:
                if user_move_name != "none":
                    computer_move_name = choice(['rock', 'paper', 'scissors'])
                    winner = calculate_winner(user_move_name, computer_move_name)

                    # Update the score based on the winner
                    if winner == "User":
                        score['wins'] += 1
                    elif winner == "Computer":
                        score['losses'] += 1
                    else:
                        score['ties'] += 1

                    # Display the current score using cv2.putText with adjusted font size and position
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, "Score: Wins - {}, Losses - {}, Ties - {}".format(score['wins'], score['losses'], score['ties']),
                                (20, 330), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                    # Display computer's move icon
                    icon = cv2.imread("images/{}.png".format(computer_move_name))
                    icon = cv2.resize(icon, (200, 200))  # Adjust the size as needed
                    frame[70:270, 400:600] = icon

                    # Display the frame
                    cv2.imshow("Rock Paper Scissors", frame)

        prev_move = user_move_name

        # display the information
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Your Move: " + user_move_name,
                    (20, 50), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Computer's Move: " + "Waiting...",
                    (350, 50), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow("Rock Paper Scissors", frame)

        k = cv2.waitKey(10)
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
