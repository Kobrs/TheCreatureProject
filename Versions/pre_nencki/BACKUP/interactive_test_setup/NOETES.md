# Goals and current state
For now I want to simulate most basic touch perception architecture - the one from  the paper "The neural circuit for touch sensivity in caenorhabditis elegans". It consists of three mechanosensory neuron groups, four inteneurons and two groups of motor neurons(B and A - basically one is responsible for forward movement and second for backward)
In this model I will ignore neuromodulators and focus only on connections weights, but neuromodulators are somehting what I want to include in my models on the next iterations.

# Implementation of basic architecture
* First will be sensory neurons: they won't differ too much from standard neurons. I'll google for characteristic parameters of those neurons and try to make them as similar to biological equivalents as possible, but I'm not gonna waste to myuch time on it.
* **The same goes for interneurons**
* **...and motor neurons** - those will be devided into two groups of 6(I guess, google it) each - they will be equivalents of forward and backward movement.