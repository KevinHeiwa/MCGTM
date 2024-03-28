_global_dict = {}
_global_dict["water_round"] = 1
_global_dict["is_ready_new_round"] = False

def set_value(key, value):
    _global_dict[key] = value

def get_value(key):
    try:
        return _global_dict[key]
    except:
        return None
    
def get_lastest_tele_count():
    return get_value("latest_tele_count")

def set_lastest_tele_count(tele_count):
    set_value("latest_tele_count", tele_count)

def reset_water_round():
    set_value("water_round", 1)

def get_water_round():
    return get_value("water_round")

def new_water_round():
    water_round = get_water_round() + 1
    set_value("water_round", water_round)
    reset_ready_new_round()


def set_ready_new_round():
    set_value("is_ready_new_round", True)

def reset_ready_new_round():
    set_value("is_ready_new_round", False)

def get_ready_new_round():
    return get_value("is_ready_new_round")

First_watermark_token = {}

second_watermark_token = []

