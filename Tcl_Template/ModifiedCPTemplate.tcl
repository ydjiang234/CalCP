wipe;
model BasicBuilder -ndm 2 -ndf 3;

#Please use mm and N as unit


set E {{E}};
set A {{A}};
set I {{I}};
set revE {{revE}};
set ampFactor {{ampFactor}};
set transfTag 1;
set outname "{{outname}}";
set d_incr {{d_incr}};
set DispList [list {{DispList}}];

#Node & Boundary
node 1 0.0 0.0;
node 2 0.0 0.0;
fix 1 1 1 1;

#Material
uniaxialMaterial ModIMKPeakOriented 1 {{CP_CMDLine}};
uniaxialMaterial Elastic 2 $revE;
uniaxialMaterial Parallel 3 1 2 -factors 1.0 -1.0;
uniaxialMaterial Elastic 4 [expr $E * $A * $ampFactor];
#Section
section Aggregator 1 4 P 4 Vy 3 Mz;
#Element
element zeroLengthSection 1 1 2 1;

pattern Plain 1 Linear {
    load 2 0.0 0.0 1.0;
}


recorder Node -file "${outname}_moment.out" -node 1 -dof 3 reaction;
recorder Node -file "${outname}_rotation.out" -node 2 -dof 3 disp;


source "D:/Git/OPS/Shared_Proc/Analysis.tcl"
set AlgOrder [list Newton NewtonLineSearch]
constraints Plain;
numberer Plain;
system BandGeneral;
set isFinish [Analyse_Static_Disp_Cyclic_Control 2 3 $DispList $d_incr 1.0E-5 50 $AlgOrder]
#print -node 1;
