## Project Overview: GPU, CPU, and MPI Acceleration in Computational Simulations

This project focuses on the application of GPU, CPU, and MPI (Message Passing Interface) techniques to enhance performance in various computational simulations and signal processing tasks. Through a series of exercises, we explored methods to optimize computations, especially in the context of stochastic processes, cellular automata, and neural network calculations. The primary objectives of this project are to improve computational efficiency, reduce execution time, and demonstrate the advantages of parallel processing in handling complex simulations.

### Tasks Completed:

1. **Zero Suppression**: Developed a GPU-accelerated function for zero suppression in waveform data, effectively eliminating low-amplitude noise. This process enhanced the clarity of waveforms and facilitated more accurate analyses.

2. **Memory Movement Optimization**: Streamlined data transfers between the CPU and GPU to reduce unnecessary roundtrips. This optimization was applied in generating pulse data and adding noise, leading to improved execution speed and resource utilization.

3. **Neural Network Acceleration**: Refactored code to enable GPU acceleration for the calculations involved in creating a hidden layer in a neural network. This included normalizing grayscale values, applying weights, and executing activation functions while ensuring the correctness of computations.

4. **Monte Carlo Stochastic Simulations**: Implemented various Monte Carlo methods to analyze stochastic processes. This included simulations that utilized random sampling techniques to estimate mathematical functions and model complex systems.

5. **Game of Life Simulation**: Developed a distributed implementation of Conway's Game of Life using MPI. In this exercise, we managed a 2D grid topology where each process calculated the next state of cells based on specific rules governing cell survival and reproduction. We utilized `MPI_Cart_create` for grid topology management and `MPI_Cart_shift` for exchanging boundary information, allowing for accurate state transitions at the edges of the grid. The simulation was designed to visualize the grid evolution over a specified number of generations.

Through this comprehensive exploration, we demonstrated the practical benefits of leveraging GPU, CPU, and MPI capabilities for significant improvements in computational speed and efficiency across various data-intensive applications. The techniques and methodologies explored in this project lay the groundwork for future advancements in high-performance computing and simulations.

This project not only highlights the advantages of parallel processing but also emphasizes the versatility of these computational techniques in addressing complex challenges in science and engineering.
