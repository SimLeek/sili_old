## mask (local gen)
### generation
(All on GPU)
* assign random values to 1024 entries within an array
* bitonic sort the 1024 entries
  * percent is x*Local_Size
* send top x/1024 to next parts of csr op
  * store full sections in 1024x smaller array
* send values as random between zero(inclusive) and one(non-inclusive)
### update
(All on GPU)
* add y to all values in csr
* remove all values >=1, store amount removed as z
* for z new values, assign rand values to x entries in an array of size 1024*x to find which workgroups to assign to (if not full)
* for all workgroups assigned to, bitonic sort random assign again
### apply
* dot product the matrix with the sin(pi*val)

* attempt RMerge with same mask in intermediate stages
* check if masked intermediate RMerge works the same as not masking the intermediate stages
* Try non-masked intermediate first and check correct to make sure masking works first