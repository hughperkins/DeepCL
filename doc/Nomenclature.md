# Nomenclature

Since recently, started approximately following the nomenclature used by [Torch neural network package](http://torch.cogbits.com/doc/nn/index.html), since the names appear to me to be:
* concise
* simple
* easy to understand

Concretely, the names for some things, nouns are:
* input: the input to the layer, coming from previous layer
* output: the output from the layer, going to the next layer
* gradInput: the partial gradient of the loss with respect to the input to the layer
* gradOutput: the partial gradient of the loss with respect to the output of the layer
* gradWeights: partial gradient of the loss with respect to the weights (renaming not done yet)

Some verbs are:
* forward: calculate the output, given the input
* backward: calculate gradInput, given gradOutput

(Note: these names are a work in progress, there are still plenty of old names about, like:
* errorsForUpstream => gradInput
* outputFromUpstream => input
* errors => gradOutput
* propagate => forward
* backprop => backward
* etc ...
)

