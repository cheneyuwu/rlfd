# Undergrad Research Project
Meeting Summary<br>
March 26th, 2019

## TODO for this weeks meeting
- Goal/Thoughts/Idea
    1. Proof required: $P(E[Q]|D, s, a)$ is normally distributed?
    2. Find environments that could benefit from having a demonstration nn that outputs the expected Q value.
    3. Use $wQ_D + (1-w)Q_E$ instead of a hard stop.
- Experiment
    1. Tests for 2 above.

## Topics for the meeting
- Proof required: $P(E[Q]|D, s, a)$ is normally distributed?
    - Refer to KeyNotes.md
- Find environments that could benefit from q estimation
    - Refer to archived data: M26Reach2DNReward/M26Reach2DPReward and Report.md for experimental results and analysis.
- Use $wQ_D + (1-w)Q_E$ instead of a hard stop.
    - Have not done this yet. However, based on analysis above, this trick probably won't work as well. We should discuss this before moving on.
- Propose the new idea of having an network trained only from the demonstration. The network will be used to help for exploration.
    - Refer to KeyNotes.md

## Reminder
