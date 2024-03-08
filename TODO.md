# todo

* convs
  * Get 1x3 and 3x1 convs working in another example
  * get sparsity loss working in an example
* csp
  * Once center of example nn is sparse, convert to compressed sparse pyramid and output
  * csp = pyramid_indices, row_indices, row values, pyramid_structure_info
  * modify displayarray shader to display csp
* contrib
  * Huge Optimization: backprop contrib as well as error. That way you don't have to save input buffers.
    * Note: this will change all conrtib calculations to look closer to error calculations, except the end.
    * Currently, we were using the direct synapse contribution vs the synapses correlation with the output error. It's easier to calculate the synapses correlation with the output vs the synapses correlation with the error. Also, while we can get the synapses direct contribution to the output, we can't get its direct contribution to the error, since that doesn't make sense. Example: 10 ppl were supposed to do 10 hours of work, but all did 0? What is each person's direct contribution to the error? Zero? Ten? One? Infinite? the amount of more work they should do is infinite. The amount each person should do for equal work is 1. If 1 person did 10 hours though, that'd mean zero error for the others. It doesn't seem easily calculable. However, using the gradient works, even when the current value is zero. And so, to compare apples to apples, we should similarly compare the output gradient to the error gradient, which is equivalent to the trivial case I used mathematically anyway.
    * Also, consider two input neurons, one has no input, one is always on or negative, both have zero mutual information. output grad would increase the connection strength between the input and output of the large input neuron while grad set its value to zero, effectively saying "stfu". Meanwhile, getting from direction contribution would allow that bad input to grow a synapse again and again.
    * Even in the sparse case, we should keep the 'strongly zero' for synaptogenesis/pruning, but not put any value in the csr
    * Effectively, we have two comparisons: 
      * mutual information between the ~~error~~ output correctness and the actual contribution of a synapse, and 
      * mutual information between the ~~error~~ output correctness and the potential contribution of a synapse
    * Now, sparsification is a matter of both value and connection strength
  * Alt Huge optimization: seperate out contrib calculation into its own kernel to run alongside forward pass.
  * (We can use the same buffer in multiple ops for huge memory reduction, but then contrib can only be calculated in the forward pass, because afterwards the input buffer is reused and modified)
  * Final optimization: use one large buffer for all images and just restrict global/max size
* Add noise to loss or optim function (done)
  * if the input & actual is perfect, the gradient will always push the synapses towards what they should be, but never make them exact
  * However, adding noise will put the synapses above and below the perfect values. Then, noise can be reduced until synapses are closer than otherwise possible.
  * (synapse noise may be modulated by connection strength)
* Allow modules to take input/output/backprop/backout buffers as input. 
  * This will allow for 'double buffering' and requiring one buffer for a whole network, instead of doubling the size. A huge performance increase when training models.
  * Should throw an error if the buffer is under the required size, but not if it's over.
  * This also means using the buffer's size to determine op size should be replaced
* Move out standard non-sparse code into its own library: DeCeIL (Device Centric Intelligence Library)
  * Convert conv backwards to not use sparse max reduce, just be slow
  * take out pyramid ops