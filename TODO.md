# todo

* convs
  * Get 1x3 and 3x1 convs working in another example
  * get sparsity loss working in an example
* code
  * create a base class for the modules
* Move out standard non-sparse code into its own library: DeCeIL (Device Centric Intelligence Library)
  * Convert conv backwards to not use sparse max reduce, just be slow
  * take out pyramid ops