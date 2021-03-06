# The Creature Project
This project aims to create environment for creating virtual creatures driven by biological neural networks simulations and use it to study different architectures, setups and their effect on virtual creature behvatiors. It is built on top of [NEURON simulator][neuron_link], [pygame][pygame_link], [pymunk][pymunk_link] and supports Lego mindstorm NXT 2.0 and EV3 bluetooth direct commands to drive the robot.

## Requirements
* [NEURON][neuron_link]
* [Pygame][pygame_link]
* [Pymunk][pymunk_link]
* [Ascii_graph][ascii_graph_link]
* Matplotlib
* Numpy
* [DEAP][deap_link] (optional)

## Installation
To use this code you should either add TheCreatureProject directory to PYTHONPATH vairable
``` bash
echo "export PYTHONPATH=$PYTHONPATH:/path/to/TheCreatureProject" >> ~/.bashrc
```
or put this TheCretureProject code to python site-packages directory, copy and run cases from somewhere else.

## Usage
The most interesting part is in Cases directory, where code for using specific architectures, parameters, models is used.


[neuron_link]: https://neuron.yale.edu/neuron/
[pygame_link]: https://github.com/pygame/pygame
[pymunk_link]: http://www.pymunk.org/en/latest/
[ascii_graph_link]:https://pypi.python.org/pypi/ascii_graph
[deap_link]: https://github.com/DEAP/deap