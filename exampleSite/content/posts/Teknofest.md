---
title: "TEKNOFEST 2022 — Deep Q-Learning for Traffic Signal Control"
date: 2022-05-28
description: "Built a reinforcement learning model using Deep Q-Learning to optimize traffic signal timing at intersections, presented at TEKNOFEST Azerbaijan's Smart Qarabag Hackathon."
tags:
  - Python
  - Reinforcement Learning
  - Deep Q-Learning
  - Hackathon
showDate: true
showReadingTime: true
showAuthor: true
showPagination: true
showTableOfContents: false
---

In May 2022, our team competed among 40 teams at the **Smart Qarabag Hackathon**, part of the TEKNOFEST Azerbaijan international technology festival. We built a reinforcement learning model for intelligent traffic signal control.

## What we built

The core idea: treat a traffic intersection as an environment and the signal controller as an agent. We used **Deep Q-Learning** to train the agent to optimize signal timing — learning when to switch lights based on real-time queue lengths, waiting times, and traffic density across all directions.

The model controlled a single intersection, adjusting green/red phase durations to minimize average vehicle waiting time and maximize throughput. Rather than following fixed-cycle timers (which is what most real intersections still do), the agent learned adaptive policies that responded to actual traffic conditions.

Our longer-term vision was multi-intersection coordination — having neighboring intersection agents communicate to achieve optimal traffic flow across a network, not just locally. We scoped the hackathon demo to a single intersection to keep it demonstrable, with the coordination layer as a clear next step.

## What came out of it

The project caught the attention of an AI specialist at the event who offered us training and mentorship to develop the system further. It was a good validation that the approach had real-world potential beyond the hackathon setting.

This was also my first hands-on project with reinforcement learning. The gap between "understanding Q-learning from a textbook" and "making an agent actually converge on a useful policy in 48 hours" was humbling — reward shaping and state representation turned out to matter far more than the choice of algorithm.

---

- **Event:** TEKNOFEST Azerbaijan 2022 — Smart Qarabag Hackathon
- **Scale:** 40 teams
- **Tech:** Deep Q-Learning, Python

*May 2022 — Baku, Azerbaijan*