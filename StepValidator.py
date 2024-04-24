def step7_validator():
    """
    StepValidator for step 7: using pedal lockring wrench to install pedal
    objects of interest in view:
        - hand(s)
        - pedal wrench
        - crank arm
        - pedal
        - bolt
    """

    """
    Initial stage condition to be satisfied (i.e. condition for starting stage):
        - pedal is in (proximity) of crank arm: pedal intersects with crank arm (can be off by a confidence threshold)
        - bolt is in crank arm: bolt bbox is fully within crank arm bbox (no reason why it shouldn't, given camera angle)
    """

    """
    Conditions to be constantly satisfied (no regression!!):
        - pedal is in (proximity, lower left) of crank arm
        - bolt is in crank arm
    """

    """
    Concerns:
        - losing visual (can use EWMA on 4 point of bbox to approximate location until next update?)
    """

    """
    In-progress conditions to be satisfied:
        1. hand intersecting greatly with pedal wrench (for the duration of rotation being sensed on sensors)
            - as long as pedal wrench+hand combination is engaged with pedal, this sub step is activated 
        2. 
    """

