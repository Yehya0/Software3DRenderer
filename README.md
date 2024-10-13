# 🖥️ Software3DRenderer

**Software3DRenderer** is a simple terminal-based 3D ASCII renderer written in C++. It demonstrates basic 3D rendering concepts such as matrix transformations, lighting models, and particle effects—visualized entirely with ASCII characters. It's a fun experiment in creating 3D visuals without traditional graphics libraries.

---

## 🎬 Demo GIF

Here’s a quick demo of the 3D renderer in action:


https://github.com/user-attachments/assets/1e93afe1-9ebb-4321-9a95-568feda1ba0e

## ⚙️ Features

- **3D ASCII Rendering**: Renders 3D objects using ASCII characters.
- **Basic Lighting Model**: Includes simple diffuse and specular lighting effects.
- **Particle Effects**: Adds dynamic particles that fade and move.
- **Matrix Transformations**: Supports 3D rotations, scaling, and projections.
- **Real-Time Rendering**: Continuously animates the scene in the terminal.

---

## 🚀 Getting Started

### Prerequisites

- A C++ compiler that supports **C++11** or later.
- Terminal/command prompt access to run the program.

### Cloning the Repository


git clone https://github.com/YourUsername/Software3DRenderer.git
cd Software3DRenderer

Building the Project
Windows (Visual Studio)

    Open the solution file (Software3DRenderer.sln) in Visual Studio.
    Build the project using Ctrl+Shift+B or Build > Build Solution.
    Run the executable (press F5 or run the .exe).

Linux/macOS (GCC/Clang)

    Open a terminal and navigate to the project directory.
    Compile the code:

    bash

g++ -std=c++11 -o Software3DRenderer main.cpp

Run the program:

bash

    ./Software3DRenderer

🛠️ How It Works

The project uses a combination of 3D vector math, matrix operations, and ASCII rendering to simulate 3D objects in the terminal. Here’s how it’s structured:

    Vec3 and Mat4 Classes: Perform vector and matrix operations like transformations, rotations, and projections.
    Lighting Model: A basic lighting system calculates how light reflects off objects using diffuse and specular reflections.
    Particles: Simulated particles are rendered with movement and fading effects.
    ASCII Rendering: 3D coordinates are converted into 2D, and ASCII characters represent the 3D shapes.

🎮 Customization

Feel free to modify the project by adjusting parameters in the code, such as:

    Rotation Speed: Change the rotation speed by adjusting the rotation variable.
    Particle Effects: Customize particle behavior in the updateParticles() function.
    Lighting Model: Tweak the drawTriangle() function to experiment with lighting effects.

🗺️ Future Plans

Potential improvements for future versions:

    Shadow Casting: Implement shadows to give objects more depth.
    Additional Shapes: Add more 3D shapes (spheres, pyramids, etc.).
    Interactive Controls: Add user interaction for moving objects or controlling camera angles.

🤝 Contributions

Contributions are welcome! Feel free to submit issues, suggest improvements, or create pull requests. Let's improve this together.


