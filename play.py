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

    return "Unknown"

def display_start_game(frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Starting Game", (300, 300), font, 2, (0, 255, 0), 4, cv2.LINE_AA)
    cv2.imshow("Rock Paper Scissors", frame)
    cv2.waitKey(1000)

def display_end_game(frame, score):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Game Over", (300, 300), font, 2, (0, 0, 255), 4, cv2.LINE_AA)
    cv2.putText(frame, f"Score: {score['wins']} wins, {score['losses']} losses, {score['ties']} ties",
                (150, 400), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Rock Paper Scissors", frame)
    cv2.waitKey(3000)

model = load_model("rock-paper-scissors-model.h5")

cap = cv2.VideoCapture(0)

prev_move = None
score = {'wins': 0, 'losses': 0, 'ties': 0}
total_games = 0
max_games_per_session = 5  # Adjust as needed
game_in_progress = False

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    if not game_in_progress:
        display_start_game(frame)
        game_in_progress = True

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
            winner = calculate_winner(user_move_name, computer_move_name)

            # Display computer's move for a short time
            cv2.putText(frame, f"Computer's Move: {computer_move_name}",
                        (500, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Rock Paper Scissors", frame)
            cv2.waitKey(1500)  # Adjust the time to display the computer's move

            if winner == "User":
                score['wins'] += 1
            elif winner == "Computer":
                score['losses'] += 1
            else:
                score['ties'] += 1

            total_games += 1
            if total_games == max_games_per_session:
                display_end_game(frame, score)
                break

            game_in_progress = False

    prev_move = user_move_name

    # display the information
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Your Move: " + user_move_name,
                (50, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Winner: " + winner,
                (400, 600), font, 2, (0, 0, 255), 4, cv2.LINE_AA)

    cv2.imshow("Rock Paper Scissors", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
