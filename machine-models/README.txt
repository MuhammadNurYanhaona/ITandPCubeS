Instructions on the format of a PCubeS description file----------------------------------------------------------

Note that the current version of the compilers do not exploit all the PCubeS features described in our technical 
report 'The Partitioned Parallel Processing Spaces (PCubeS) Type Architecture' that would be needed to generate 
fully optimized executables. Our plan is to incorporate more and more PCubeS features in the compilers' code 
generation logic with time. Therefore, the textual PCubeS description one needs to write at this moment to 
compile his code for a target hardware is a much shorter version of its actual PCubeS model. Our advice is one 
should keep currently unused features listed as comments so that he can enhance his PCubeS model easily in the 
future as later version of the compilers start using them.

Further, note that the user should investigate the complete PCubeS description of a target, i.e, all features,
to make the right choice about the LPU sizes and degrees of parallelism in his program. It is just the compilers
are not capable enough to further assist him in gaining even better performance over his good decisions at this
time. 
 
Syntax of the PCubeS file----------------------------------------------------------------------------------------

1. '//' indicates the beginning of a single line comment in the description

2. The PCubeS spaces (PPSes) should be listed in a top down manner, i.e., the higher spaces should come before 
   the lower spaces; each PPS should have its own line and comments and empty lines can appear in-between two PPS
   lines.

3. Currently a PPS line is described using the following syntax.
	
	'Space' $spaceNo $attributes ':' $spaceName '('$ppuCount')'
    
   Here the attributes are optional and, when present, listed as one after another using the following syntax
	
	'<' $attr1Name '>' '<' $attr2Name '>' ...

   Currently there are only three attributes used by the compilers
	
	a) core: this attribute tells from what level physical core numbering starts in the hardware. The core 
        numbering does not start necessarily from the lowest PPS in all cases. For example, if the PCubeS model
	wants to expose hyperthreads of a core. Therefore, we needed this attribute to process the information
	of the core numbering file currectly.

	b) segment: this attribute tells that the PPS having the attribute is the level where memory 
	segmentation begins in the machine. It is only used by the segmented-memory backend compiler. Remember
	that memory segmentation at a PPS means that the PPUs of that PPS have separate memories. At the time
	of this writing, multiple segmentation levels is not understood by the segmented-memory compiler.

	c) unit: this attribute tells where hardware breakdown happens in the PCubeS hierarchy. In other word,
	all PPSes below this level resides in the confinement of a single hardware unit, i.e, a CPU. This 
	attribute is only used by the segmented-memory compiler as multicore targets have only one physical unit.

4.  Finally, one should describe the machine as a rooted hierarchy, i.e., the top-most PPS should have only one 
    PPU

An example PCubeS description (Hermes Cluster, CS, UVA)----------------------------------------------------------	      

//---------------------------------------------------------------------------------------------------------------
//Space #Number :               $Space-Name     (#PPU-Count)    // Comments
//---------------------------------------------------------------------------------------------------------------
Space   6:                      Cluster         (1)		// nodes are connected by 10 GB ethernet
Space   5<unit>:                Node            (4)             // there are four nodes 
Space   4<segment>:             CPU             (4)             // 64 GB RAM Per CPU
Space   3:                      NUMA-Node       (2)             // 6 MB L-3 Cache
Space   2:                      Core-Pair       (4)             // 2 MB L-2 Cache (1 floating point ALU unit)
Space   1<core>:                Core            (2)             // 16 KB L-1 Cache (1 integer ALU unit)

