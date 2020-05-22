#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 12:34:02 2020

@author: arko
"""

from keras.models import load_model
import cv2
import numpy as np
from random import choice

REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissor",
    3: "lizard",
    4: "spock",
    5: "none"
}


def mapper(val):
    return REV_CLASS_MAP[val]


def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"

    if move1 == "rock":
        if move2 == "scissor":
            return "User"
        if move2 == "paper":
            return "Computer"
        if move2 == "lizard":
            return "User"
        if move2 == "spock":
            return "Computer"

    if move1 == "paper":
        if move2 == "rock":
            return "User"
        if move2 == "scissor":
            return "Computer"
        if move2 == "spock":
            return "User"
        if move2 == "lizard":
            return "Computer"

    if move1 == "scissor":
        if move2 == "paper":
            return "User"
        if move2 == "rock":
            return "Computer"
        if move2 == "lizard":
            return "User"
        if move2 == "spock":
            return "Computer"
        
    if move1 == "lizard":
        if move2 == "paper":
            return "User"
        if move2 == "rock":
            return "Computer"
        if move2 == "spock":
            return "User"
        if move2 == "scissor":
            return "Computer"
    
    if move1 == "spock":
        if move2 == "rock":
            return "User"
        if move2 == "lizard":
            return "Computer"
        if move2 == "scissor":
            return "User"
        if move2 == "paper":
            return "Computer"
        

model = load_model("rock-paper-game.h5")

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

prev_move = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    scale_percent = 100 # percent of original size
    width = 1500
    height = 1080
    dim = (width, height)
    
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
   # cv2.resize(frame,(1920,1080),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)
    # rectangle for computer to play
    cv2.rectangle(frame, (800, 100), (1200, 500), (255, 255, 255), 2)

    # extract the region of image within the user rectangle
    roi = frame[100:500, 100:500]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (227, 227))

    # predict the move made
    pred = model.predict(np.array([img]))
    move_code = np.argmax(pred[0])
    user_move_name = mapper(move_code)

    # predict the winner (human vs computer)
    if prev_move != user_move_name:
        if user_move_name != "none":
            computer_move_name = choice(['rock', 'paper', 'scissor', 'lizard', 'spock'])
            winner = calculate_winner(user_move_name, computer_move_name)
        else:
            computer_move_name = "none"
            winner = "Waiting..."
    prev_move = user_move_name

    # display the information
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Your Move: " + user_move_name,
                (50, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Computer's Move: " + computer_move_name,
                (750, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Winner: " + winner,
                (400, 600), font, 2, (0, 0, 255), 4, cv2.LINE_AA)

    if computer_move_name != "none":
        icon = cv2.imread(
            "images/{}.png".format(computer_move_name))
        icon = cv2.resize(icon, (400, 400))
        frame[100:500, 800:1200] = icon

    cv2.imshow("Rock Paper scissor", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()