# todo

* conv reverse
  * Sum reduce conv instead of max reduce
  * (done) pre-compute reduction points and give index list to next reduce shader (pyr.size*pyr.levels/1024+a few)
  * determine input to initial and next reductions based on pre-computed index list
  * reduce to find lowest next index during reduction, if any
  * output sum reduction at global index where local_index==0
  * if no next index, continue, else sum reduce again
  * loop prev until no next index
  * run similar algo until all reduction points are done at pyramid i/o indices
  * modify backwards op to select from pyramid io points / workgroup size to conv err
    * divide error sum for each `in*lvl+out` location by out_resolution (`out_width*out_height*channels`)
* Move out standard non-sparse code into its own library: DeCeIL (Device Centric Intelligence Library)
  * Convert conv backwards to not use sparse max reduce, just be slow
  * take out pyramid ops