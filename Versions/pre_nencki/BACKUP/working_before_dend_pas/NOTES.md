# Goals and current state
For now I want to simulate most basic touch perception architecture - the one from  the paper "The neural circuit for touch sensivity in caenorhabditis elegans". It consists of three mechanosensory neuron groups, four inteneurons and two groups of motor neurons(B and A - basically one is responsible for forward movement and second for backward)
In this model I will ignore neuromodulators and focus only on connections weights, but neuromodulators are somehting what I want to include in my models on the next iterations.

Basic static touch circuit is defined as the on presented in mentioned paper, but without LUA neuron (I don't want to deal with gap junctions just yet) and instead one additional chemical synapse vrom AVA and AVD to PLM.
Every gap junction will be replaced by two opposite chemical synapse if this connection doesn't already exist.

Basic static touch circuit have 

# Implementation of basic architecture
* First will be sensory neurons: they won't differ too much from standard neurons. I'll google for characteristic parameters of those neurons and try to make them as similar to biological equivalents as possible, but I'm not gonna waste to myuch time on it.
* **The same goes for interneurons**
* **...and motor neurons** - those will be devided into two groups of 6(I guess, google it) each - they will be equivalents of forward and backward movement.
* **GA implementation** My current idea of best approach is follwoing: don't code desired architecture directly! It makes no sens, because I'm going to use GAs anyway, so this architecture would go to waste in the moment I'll begin impleemnting GA. That being said, I should firstly define DNA schema, functions and basic plan of implementaion, implement decoding mechanisms, manually encode architecture in DNA, check if everything is exactly as it's supposed to be and then start implementation of core GA functionality.
* **GA DNA structure** 
*Deprecated idea!!!*
The stucture of DNA will be divided into two basic parts:
 + Cell specification
 + Connection specification
*Each of those will be captured in data frames. How bits are interpreted depends on the frame type. There will be two frame identifiers, each will contain 4 bits. Algorithm will first search DNA for frame identifiers. Frames will not be closed. This mean that after frame identifier is detected next x bits will be read and interpreted in specified way and then searh for frames will continue beginning from x+1 bits after last bit of opening frame.*

*New idea*
Simply split data into frames, which are opened and closed, in ech one encdoe cell. Start from properties and after them all data till the end will be connections specifications(16 bit weight and 4 bit delay). If frame is no longer than length of cell info + one connection, then the frame is ignored.

*Core implementatoion rules*:
+ We allow frame codon to both close one frame and open next frame.
+ Connections are only possible between defined cells - if cell isn't defined this means that it doesn't have any connections to other cells, so it's virtually useless(motor neurons would be exception but easy to overcome for GA by createing connection with almost 0 weight). This makes things simpler and cleaner.
+ Ok, I think that besides weight, w_type and delay I should also include passive current into the basic implementaiton, as it's basically essential to allow for spiking without any stimuli.


**NOTE**: *Current implementataion uses VecStim artifical cell, as I wasn't able to make IClamp working correctly(it aplied only first spike) and due to time restriction the VecStim will be sufficient*


#WORKPLACE
Individual for fixed architecture
* this determines whether connection should be inhibitory or excitatory (it isn't direcly weight sign, it can only be interpreted this way. Rememeber that in  current setup [non artificial cells] weights can NEVER BE NEGATIVE!). 0 stands for inhibitory synapse and 1 for excitatory.
111111 00000001 10000000 01001000 00000010 1 000010011010000 0010 111111
fo   cell_id  dend_len   pas_curr target   *  weight           d    fc
     1        128        -55                     0.4           2

1111110000000110000000010010000000001010000100110100000010111111

connections:
0 -> 7 2 4
1 -> 2 3 4
2 -> 7 5 3 1
3 -> 2 1 6
4 -> 3 0 1 6
5 -> 7
6 -> 3
7 -> 5 3 6

Therefore we have 22 connections, each of which in basic version is 20 bit long (16 for weight and 4 for delay), therefore we have 440 bits of DNA for basic implementation.

Static network architecture:
{0: {'connections': [(7, 0.4, 2), (2, 0.4, 2), (4, 0.4, 2)], 'dend_len': 128, 'dend_pas': -55}, 1: {'connections': [(2, 0.4, 2), (3, 0.4, 2), (4, 0.4, 2)], 'dend_len': 128, 'dend_pas': -55}, 2: {'connections': [(7, 0.4, 2), (5, 0.4, 2), (3, 0.4, 2), (1, 0.4, 2)], 'dend_len': 128, 'dend_pas': -55}, 3: {'connections': [(2, 0.4, 2), (1, 0.4, 2), (6, 0.4, 2)], 'dend_len': 128, 'dend_pas': -55}, 4: {'connections': [(3, 0.4, 2), (0, 0.4, 2), (1, 0.4, 2), (6, 0.4, 2)], 'dend_len': 128, 'dend_pas': -55}, 5: {'connections': [(7, 0.4, 2)], 'dend_len': 128, 'dend_pas': -55}, 6: {'connections': [(3, 0.4, 2)], 'dend_len': 128, 'dend_pas': -55}, 7: {'connections': [(5, 0.4, 2), (3, 0.4, 2), (6, 0.4, 2)], 'dend_len': 128, 'dend_pas': -55}}


Best specimens:
Evaluating specimen:
10000100110100000010100001001101000010111000010011110000001010000100111100000010100001001101000000101001010011010000001100000100110100000010100001001101000000101000010011010000001010000100110100000010100001011101000000101000010011010000001010001100110100000010110001001101000000100000010011010000001010100100110100000010100001001101000000101000010011010000001010000100110110000010100001001101000010101000010011010000001010000100110100000010
{0: {'connections': [(7, (1, 0.1232), 2), (2, (1, 0.1232), 11), (4, (1, 0.1264), 2)], 'dend_len': 128, 'dend_pas': -55}, 1: {'connections': [(2, (1, 0.1264), 2), (3, (1, 0.1232), 2), (4, (1, 0.5328), 3)], 'dend_len': 128, 'dend_pas': -55}, 2: {'connections': [(7, (0, 0.1232), 2), (5, (1, 0.1232), 2), (3, (1, 0.1232), 2), (1, (1, 0.1232), 2)], 'dend_len': 128, 'dend_pas': -55}, 3: {'connections': [(2, (1, 0.1488), 2), (1, (1, 0.1232), 2), (6, (1, 0.328), 2)], 'dend_len': 128, 'dend_pas': -55}, 4: {'connections': [(3, (1, 1.7616), 2), (0, (0, 0.1232), 2), (1, (1, 0.9424), 2), (6, (1, 0.1232), 2)], 'dend_len': 128, 'dend_pas': -55}, 5: {'connections': [(7, (1, 0.1232), 2)], 'dend_len': 128, 'dend_pas': -55}, 6: {'connections': [(3, (1, 0.124), 2)], 'dend_len': 128, 'dend_pas': -55}, 7: {'connections': [(5, (1, 0.1232), 10), (3, (1, 0.1232), 2), (6, (1, 0.1232), 2)], 'dend_len': 128, 'dend_pas': -55}}
Score: (28.387519999999977,)


Evaluating specimen:
00000100110100000010100001001101000000101000010011010000001010000100110100000010100000001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000010000001001101000000101000010011010000001010000100110100000011100000001101000000101000010011010000101010000100110100000010100001001101000000101000010011010000101010000100110100000010100001001101000000101000010011010000001010000100010100000010
{0: {'connections': [(7, (0, 0.1232), 2), (2, (1, 0.1232), 2), (4, (1, 0.1232), 2)], 'dend_len': 128, 'dend_pas': -55}, 1: {'connections': [(2, (1, 0.1232), 2), (3, (1, 0.0208), 2), (4, (1, 0.1232), 2)], 'dend_len': 128, 'dend_pas': -55}, 2: {'connections': [(7, (1, 0.1232), 2), (5, (1, 0.1232), 2), (3, (1, 0.1232), 2), (1, (1, 0.1232), 2)], 'dend_len': 128, 'dend_pas': -55}, 3: {'connections': [(2, (0, 0.1232), 2), (1, (1, 0.1232), 2), (6, (1, 0.1232), 3)], 'dend_len': 128, 'dend_pas': -55}, 4: {'connections': [(3, (1, 0.0208), 2), (0, (1, 0.1232), 10), (1, (1, 0.1232), 2), (6, (1, 0.1232), 2)], 'dend_len': 128, 'dend_pas': -55}, 5: {'connections': [(7, (1, 0.1232), 10)], 'dend_len': 128, 'dend_pas': -55}, 6: {'connections': [(3, (1, 0.1232), 2)], 'dend_len': 128, 'dend_pas': -55}, 7: {'connections': [(5, (1, 0.1232), 2), (3, (1, 0.1232), 2), (6, (1, 0.1104), 2)], 'dend_len': 128, 'dend_pas': -55}}
Score: (38.52591999999996,)


Evaluating specimen:
10000100110100000010100001001101000000101000010011010000001010000100110100000010100001001101000000100000010011010000001010000100110100100010100001001101000000101000010011010000001010000100110100000010100001001101000000101000010011010000001010000100110100000111100001001111000000101000010011010000001010000100010100000010100001001101000000101000010011010000001010000100110100000010100001001101000010101000010011010000001110000100110100000010
{0: {'connections': [(7, (1, 0.1232), 2), (2, (1, 0.1232), 2), (4, (1, 0.1232), 2)], 'dend_len': 128, 'dend_pas': -55}, 1: {'connections': [(2, (1, 0.1232), 2), (3, (1, 0.1232), 2), (4, (0, 0.1232), 2)], 'dend_len': 128, 'dend_pas': -55}, 2: {'connections': [(7, (1, 0.1234), 2), (5, (1, 0.1232), 2), (3, (1, 0.1232), 2), (1, (1, 0.1232), 2)], 'dend_len': 128, 'dend_pas': -55}, 3: {'connections': [(2, (1, 0.1232), 2), (1, (1, 0.1232), 2), (6, (1, 0.1232), 7)], 'dend_len': 128, 'dend_pas': -55}, 4: {'connections': [(3, (1, 0.1264), 2), (0, (1, 0.1232), 2), (1, (1, 0.1104), 2), (6, (1, 0.1232), 2)], 'dend_len': 128, 'dend_pas': -55}, 5: {'connections': [(7, (1, 0.1232), 2)], 'dend_len': 128, 'dend_pas': -55}, 6: {'connections': [(3, (1, 0.1232), 2)], 'dend_len': 128, 'dend_pas': -55}, 7: {'connections': [(5, (1, 0.1232), 10), (3, (1, 0.1232), 3), (6, (1, 0.1232), 2)], 'dend_len': 128, 'dend_pas': -55}}
Score: (56.829799999999636,)


