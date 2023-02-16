# %%
action_score = {
    'write': 1,
    'read': 1,
    'eat': 0.1,
}

emotion_score = {
    "Angry": 0.2,
    "Disgust": 0.2,
    "Fear": 0.2,
    "Happy": 0.5,
    "Sad": 0.2,
    "Surprise": 0.3,
    "Neutral": 1,
}

max_blink_num = 0.5
max_blink_time = 1000

def get_score(action: str, emotion: str, blink_num: int, blink_time: int):
    if action == "None" or emotion == "None" or blink_num == -1 or blink_time == -1:
        return 0
    blink_num = min(blink_num, max_blink_num)
    blink_time = min(blink_time, max_blink_time)

    return 20 * action_score[action] \
            + 10 * emotion_score[emotion] \
            + 40 * (max_blink_num - blink_num) / max_blink_num \
            + 30 * blink_time / max_blink_time


# ex
action = 'write'
emotion = 'Angry'
blink_num = 1
blink_time = 67
get_score(action, emotion, blink_num, blink_time)


