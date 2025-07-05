habit_risk_scores = {
    "non-smoker": -20,
    "smokes": 50,
    "smokes occasionally": 35,
    "smokes heavily": 120,
    "smokes rarely": 25,

    "drinks": 30,
    "drinks occasionally": 40,
    "drinks heavily": 80,

    "no exercise": 25,
    "no activity": 25,
    "no movement": 25,
    "no physical activity": 25,
    "sedentary": 30,
    "completely sedentary": 30,
    "bedridden": 50,
    "immobile": 50,
    "wheelchair": 40,
    "wheelchair-bound": 40,

    "stressed": 20,
    "desk job": 10,
    "office job": 10,

    "runs daily": -15,
    "runs 3x week": -10,
    "runner": -10,
    "walks": -4,
    "walks daily": -8,
    "walks sometimes": -3,
    "some walking": -3,
    "light walking": -3,
    "walks dog": -4,
    "swims": -6,
    "swims regularly": -6,
    "cycles": -6,
    "cycles daily": -6,
    "occasional cycling": -4,
    "hikes": -5,
    "hikes weekends": -5,

    "yoga weekly": -5,
    "yoga enthusiast": -5,
    "pilates": -5,
    "fitness class": -6,
    "exercises weekly": -5,
    "active lifestyle": -8,
    "light gardening": -3,

    "gym": -6,
    "gym member": -6,
    "gym twice a week": -6,

    "athlete": -20,
    "tennis player": -5,
    "soccer player": -10,
    "dances weekly": -4,
    "rock climbing": -8
}

def process_habits(habits_string):
    points = 0
    habits = habits_string.split(';')
    for habit in habits:
        habit = habit.strip()
        habit = habit.lower()
        try:
            points += habit_risk_scores[habit]
        except:
            print(f"Unknown habit: {habit}")
            continue
    return points / 4

def boost_habits(habits_string):
    return habits_string * 1.2
    

    