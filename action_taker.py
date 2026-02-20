
gestures = []

machine_state = None


def take_action(gesture, index_finger_coordinates, thumb_tip, time_and_gesture):
    #print(f"{gesture} Index finger location: {index_finger_coordinates.x},{index_finger_coordinates.y} Thumb tip: {thumb_tip.x}, {thumb_tip.y}")
    change_machine_state(time_and_gesture)


def are_values_same(dictionary, start_key, end_key, target_value=None):
    """
    Checks if all items between two keys in a dictionary have the same value.
    Assumes keys are ordered and present in the dictionary.
    """
    # Extract the keys in the desired range. This example assumes integer keys.
    # Adjust key retrieval logic if your keys are strings or have different ranges/ordering.
    relevant_keys = [k for k in sorted(dictionary) if start_key <= k <= end_key]

    if not relevant_keys:
        return True  # Or handle as appropriate, e.g., raise an error or return False

    first_value = dictionary[relevant_keys[0]]
    # Check if all other values in the range are equal to the first value
    if all(dictionary[key] == "Open_Palm" for key in relevant_keys[1:]):
        return "active"
    
    elif all(dictionary[key] == "Closed_Fist" for key in relevant_keys[1:]):
        return "passive"
    
    else:
        print("Different gestures")
    


def machine_active():
    print("The machine is active")

def machine_passive():
    print("The machine is passive")

def change_machine_state(time_and_gesture):
    global machine_state
    #Logic to check whether the gestures for the past two seconds have been open_palm
    # print(time_and_gesture)

    # 1. Find the key of last item in the dictionary (i.e. latest time)
    # 2. Subtract 2 seconds from it to find the index of the gesture two seconds ago
    # 3. Find out if all the gestures in those two seconds have been an open palm
    # 4. Change the machine state to passive

    latest_time = next(reversed(time_and_gesture))
    time_two_seconds_ago = latest_time - 2000

    truth = are_values_same(time_and_gesture, time_two_seconds_ago, latest_time, "Open_Palm")

    machine_state = truth
    
    print(f"Machine state is now {machine_state}")


    return 0



