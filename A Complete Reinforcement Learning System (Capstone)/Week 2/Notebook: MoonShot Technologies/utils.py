def get_velocity(action):
    if action == 1:
        return 12, 15
    if action == 2:
        return 10, 10
    if action == 3:
        return 0, 0
    if action == 4:
        return 12, 15
    if action == 5:
        return 0, 0

def get_angle(action):
    if action == 1:
        return 0
    if action == 2:
        return 15
    if action == 3:
        return 0
    if action == 4:
        return 0
    if action == 5:
        return 0

def get_position(action):
    if action == 1:
        return 10, 100
    if action == 2:
        return 50, 0
    if action == 3:
        return 50, 0
    if action == 4:
        return 50, 20
    if action == 5:
        return 2, 0

def get_landing_zone():
    return 50, 0

def get_fuel(action):
    if action == 1:
        return 10
    if action == 2:
        return 20
    if action == 3:
        return 5
    if action == 4:
        return 0
    if action == 5:
        return 10

def tests(LunarLander, test_number):
    ll = LunarLander()
    ll.env_start()
    reward, obs, term = ll.env_step(test_number)
    print("Reward: {}, Terminal: {}".format(reward, term))