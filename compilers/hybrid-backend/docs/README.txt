Things for future development
----------------------------------------------------------------------------------------------------------
1. IT array dimension ranges can start from any point in the integer line and can be decreasing
as opposed to only increasing ranges supported by contemporary languages. We handled decreasing
ranges in the multicore backend compiler. This compiler, however, does not deal with decreasing
dimension ranges. This is because we want to avoid bogged down by all these features that will 
be used rarely in programs but take some time to incorporate in the prototype compiler we are 
currently developing. Note that we are not saying that supporting decreasing ranges will make 
generated code to suffer in performance. We are just avoiding dealing with it to save more time.  
