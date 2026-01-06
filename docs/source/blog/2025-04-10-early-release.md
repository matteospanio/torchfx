---
blogpost: true
date: Apr 14, 2025
author: Matteo Spanio
category: announcements
tags: open-source, dafx, welcome
---

# 🚀 TorchFX: Audio Filters Powered by PyTorch

*An open-source academic project at the intersection of audio DSP, performance, and modern machine learning tooling.*

## 🔊 Why TorchFX?

TorchFX was born from a **very concrete research and industry need**.

During my PhD, while collaborating with an industrial partner, I needed **fast, flexible, and expressive software** to prototype **audio filters**. Existing DSP tools were powerful but often rigid, CPU-bound, or disconnected from the rapidly evolving deep learning ecosystem.

So I asked a simple question:

> *What if we built audio filters on top of PyTorch instead of NumPy / SciPy?*

TorchFX is my answer.

## ⚡ The Core Idea

The idea behind TorchFX is intentionally simple:

* **Use PyTorch as the computational backend**
* Treat audio filters as **first-class differentiable operators**
* Enable:

  * 🚀 **GPU acceleration**
  * 🧮 **Automatic differentiation**
  * 🧵 **Efficient multithreaded CPU execution**
  * 🔬 Seamless integration with deep learning workflows

Even when running on CPU, PyTorch often outperforms NumPy thanks to its **optimized multithreading**.
And while tools like **Numba** can provide performance boosts, I found them:

* harder to install and maintain across systems
* less aligned with the fast-moving deep learning ecosystem

PyTorch, instead, gives us access to a **huge and growing ecosystem** of tools, libraries, and researchers — making TorchFX future-proof by design.

## 🧠 Differentiable Audio DSP

With TorchFX, audio filters are not just fast — they are **differentiable**.

This unlocks exciting possibilities:

* Gradient-based optimization of filter parameters
* End-to-end learning systems that include classical DSP blocks
* Hybrid models combining **signal processing and neural networks**

TorchFX is designed for:

* audio researchers
* DSP practitioners
* ML engineers working with sound
* anyone interested in **bridging classic audio DSP and modern ML**

## 📄 From Research to Open Source

TorchFX is not just a library — it is also a **research contribution**.

* 📘 The accompanying paper has been **accepted at DAFx 2025**
* 🧾 A **preprint is available on arXiv**
  👉 *(link to be added)*

Releasing the preprint immediately sparked interest in the audio research community.
TorchFX even appeared among the **trending projects on Papers With Code** (RIP 💔 — but we remember).

This response confirmed something important to me:

> There is a real need for open, performant, and differentiable audio DSP tools.

## 🌱 An Open-Source Academic Project

TorchFX is proudly:

* 🧪 **academic**
* 🔓 **open source**
* 🤝 **open to contributions**

My hope is that TorchFX will grow into a shared platform for experimenting, prototyping, and researching audio filters — whether for classic DSP, machine learning, or something in between.

If you are curious, excited, or skeptical — I’d love for you to try it, break it, and improve it.

---

## 🔗 Project Links

* 📦 GitHub repository: [https://github.com/matteospanio/torchfx](https://github.com/matteospanio/torchfx)
* 📖 Documentation: [https://matteospanio.github.io/torchfx/](https://matteospanio.github.io/torchfx/)
* 🧾 arXiv preprint: [https://arxiv.org/abs/2504.08624](https://arxiv.org/abs/2504.08624)

## 🌍 Looking Ahead

TorchFX is just getting started.

The future I’m excited about is one where:

* audio DSP is **fully differentiable**
* prototypes are **fast and expressive**
* researchers and practitioners share tools, not silos

Thanks for being here at the beginning 🚀
Let’s build the future of audio processing — together.
