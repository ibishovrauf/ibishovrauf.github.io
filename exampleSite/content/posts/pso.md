---
title: "2nd Place at UFAZ Hackathon — Visualizing Particle Swarm Optimization"
date: 2022-03-01
description: "How our 3-person team built an interactive PSO visualization across four benchmark functions and placed 2nd among 40 teams at the French-Azerbaijani University hackathon."
tags:
  - Python
  - NumPy
  - Matplotlib
  - Optimization
  - Hackathon
showDate: true
showReadingTime: true
showAuthor: true
showPagination: true
showTableOfContents: true
---

In March 2022, our 3-person team placed 2nd out of 40 teams at the UFAZ (French-Azerbaijani University) Hackathon, winning a 2,500 AZN prize. We built an interactive visualization of Particle Swarm Optimization — showing how a swarm of particles explores a solution space and converges on the global minimum.

This post is about what we built, why it worked as a hackathon project, and what I learned from it.

---

## The idea

The hackathon challenge was open-ended, and most teams gravitated toward web apps or data dashboards. We went a different direction: take a concept from optimization theory that's hard to explain with words alone and make it *visible*.

Particle Swarm Optimization is one of those algorithms that's simple to describe ("particles fly around and remember the best position they've found") but hard to intuit. How does a swarm of independent particles, each following simple rules, reliably find the minimum of a complex function with dozens of local minima? The answer clicks instantly when you watch it happen.

That was the pitch: don't just implement PSO, make people *see* it.

---

## How PSO works

The core idea is borrowed from how birds flock or fish school. You scatter a population of particles across a search space. Each particle has a position and a velocity. On every iteration, each particle adjusts its velocity based on three forces:

**Inertia** — keep moving in the direction you were going. Controlled by a weight `W`. We used `W = 0.8`, which gives enough momentum to explore without overshooting.

**Personal memory** — pull toward the best position *this particle* has ever found (`pbest`). Controlled by coefficient `c1`.

**Social influence** — pull toward the best position *any particle in the swarm* has found (`gbest`). Controlled by coefficient `c2`.

The velocity update rule:

```
v_new = W * v_current + c1 * r1 * (pbest - position) + c2 * r2 * (gbest - position)
```

where `r1` and `r2` are random factors that add stochasticity — without them, particles would follow deterministic paths and get stuck in local minima.

After updating velocity, the particle moves: `position_new = position + v_new`. Then it checks if its new position is better than its personal best, and the swarm updates the global best if needed.

That's the entire algorithm. No gradients, no derivatives, no assumptions about the function being smooth or differentiable. It just works — which is exactly what makes it fascinating to watch.

---

## What we built

We implemented PSO in Python with NumPy and Matplotlib, and ran it on four classic benchmark optimization functions — each designed to be hard in a different way:

**Ackley function** — a nearly flat outer region with a deep hole at the center, surrounded by a field of smaller local minima. Tests whether particles can avoid getting trapped in the cosmetic bumps and find the true global minimum at the origin. Search domain: [-32.768, 32.768].

**Rastrigin function** — a grid of evenly-spaced local minima that looks like an egg carton. The global minimum sits at the origin, but there are `10^n` local minima to get stuck in (where n is the dimension). Search domain: [-5.12, 5.12].

**Rosenbrock function** — the "banana function." The global minimum sits inside a long, narrow, curved valley. Finding the valley is easy; converging to the minimum *within* the valley is hard. Tests patience more than exploration. Search domain: [-5, 10].

**Schwefel function** — the nastiest one. The global minimum is geometrically far from the next-best local minima, so gradient-based intuition actively misleads you. The best solution is nowhere near where you'd expect. Search domain: [-500, 500].

For each function, we ran 30 particles for 30 iterations in 2D (plotted on a 3D surface) and in higher dimensions (where we tracked convergence numerically). The 3D visualizations showed particles as black dots scattered across a semi-transparent surface plot — you could watch them cluster toward the minimum.

We also computed the swarm's mean position and standard deviation at convergence, which gave a quantitative measure of how tightly the particles had converged.

---

## The implementation

The `Particle` class is minimal — each particle tracks its position, velocity, personal best, and personal best score:

```python
class Particle:
    def __init__(self, position):
        self.position = position
        self.velocity = np.array([random.uniform(-1,1) for i in range(len(position))])
        self.pbest = position
        self.pbest_z = equation(position)
```

The velocity update follows the standard PSO formula with random scaling:

```python
def updateVelocity(self):
    self.velocity = (W * self.velocity 
                    + c1 * random.uniform(0, 0.5) * (self.pbest - self.position)
                    + c2 * random.uniform(0, 0.5) * (global_best - self.position))
```

Each benchmark function is a standalone Python function that takes a list of coordinates and returns the value — making it trivial to swap functions and compare behavior:

```python
def rastrigin(args):
    answer = 10 * len(args)
    for arg in args:
        answer += arg**2 - 10 * cos(2 * pi * arg)
    return answer
```

The visualization renders the objective function as a 3D surface with Matplotlib, overlays particle positions as scatter points, and annotates the plot with convergence statistics.

---

## What made it work as a hackathon project

Looking back, three things made this project land with the judges:

**It was immediately understandable.** We could demo the 3D visualizations and anyone — regardless of their math background — could see particles swarming toward the lowest point on a surface. The Schwefel function was particularly dramatic: particles initially scatter across a landscape of deceptive minima, then gradually converge on a solution that's far from where you'd expect.

**It scaled from simple to impressive.** The 2D/3D visualizations were visually engaging, but we could also show that the same code worked in arbitrary dimensions — where visualization breaks down but the algorithm still converges. Running Rastrigin in 10 dimensions and showing convergence numerically demonstrated that this wasn't just a pretty plot.

**It was scientifically honest.** We showed both successes and failures. PSO doesn't always find the exact global minimum — sometimes it gets stuck near a local minimum, especially on Schwefel. We showed the standard deviation of the swarm's final positions as a convergence metric, which gave the judges quantitative evidence rather than just visuals.

---

## What I learned

This was March 2022 — about six months before I joined NAIC. Looking back, this project planted seeds for things I'd use later in my career:

**Optimization intuition matters.** Understanding how search works — exploration vs. exploitation, getting stuck in local optima, the role of randomness — turned out to be directly relevant when I later worked on search ranking and retrieval systems. The trade-offs are structurally similar: explore broadly or exploit what you've already found.

**Visualization is an argument.** The strongest part of our presentation wasn't the code or the math — it was showing the particles move. I've carried this into every project since: if you can't show someone what your system is doing, you don't fully understand it yourself.

**Hackathons reward boldness.** Most teams built CRUD apps. We built something that required explaining a mathematical concept to non-technical judges — and it worked because we made the concept visible. The lesson: pick the harder project if you can make it accessible.

---

## Links

- [GitHub Repository](https://github.com/alizadasaleh/PSO)
- Event: UFAZ Hackathon 2022, French-Azerbaijani University
- Result: 2nd place / 40 teams
- Prize: 2,500 AZN

---

*March 2022 — Baku, Azerbaijan*