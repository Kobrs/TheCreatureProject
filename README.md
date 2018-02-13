# The Creature Project
This project aims to create environment for creating virtual creatures driven by biological neural networks simulations and use it to study different architectures, setups and their effect on virtual creature behvatiors. It is built on top of [NEURON simulator][neuron_link], [pygame][pygame_link], [pymunk][pymunk_link] and supports Lego mindstorm NXT 2.0 and EV3 bluetooth direct commands to drive the robot.

## Setup
To use this code you should either add TheCreatureProject directory to PYTHONPATH vairable
``` bash
echo "export PYTHONPATH=$PYTHONPATH:/path/to/TheCreatureProject" >> ~/.bashrc
```
or put this TheCretureProject code to python site-packages directory, copy and run cases from somewhere else.


[neuron_link]: https://neuron.yale.edu/neuron/
[pygame_link]: https://github.com/pygame/pygame
[pymunk_link]: http://www.pymunk.org/en/latest/