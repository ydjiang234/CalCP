wipe;
model BasicBuilder -ndm 2 -ndf 3;

#Please use mm and N as unit

set L {{L}};
set E {{E}};
set A {{A}};
set I {{I}};
set revE {{revE}};
set ampFactor {{ampFactor}};
set Naxial {{Naxial}};
set transfTag 1;
set outname "{{outname}}";
set d_incr {{d_incr}};

#Node & Boundary
node 1 0.0 0.0;
node 1111 0.0 0.0;
node 2 0.0 $L;
fix 1 1 1 1;

#Material
uniaxialMaterial ModIMKPeakOriented 1 {{CP_CMDLine}};
uniaxialMaterial Elastic 2 $revE;
uniaxialMaterial Parallel 3 1 2 -factors 1.0 -1.0;
uniaxialMaterial Elastic 4 [expr $E * $A * $ampFactor];
#Section
geomTransf Corotational $transfTag;
section Elastic 2 $E $A $I;
section Aggregator 1 4 P 4 Vy 3 Mz;
element zeroLengthSection 1 1 1111 1;
element elasticBeamColumn 2 1111 2 $A $E $I $transfTag;

#Gravity
#Gravity Loads
if {$Naxial != 0.0} {
    pattern Plain 2 Linear {
        load 2 0.0 -$Naxial 0.0;
    ;
    }
    constraints Transformation;
    numberer RCM;
    system BandGeneral;
    test NormDispIncr 1.0E-03 50;
    algorithm Newton;
    integrator LoadControl 0.1;
    analysis Static;
    analyze 10;
    loadConst -time 0.0;
    puts "Gravity loads Added";
}

pattern Plain 1 Linear {
    load 2 1.0 0.0 0.0;
}


recorder Node -file "${outname}_force.out" -node 1 -dof 3 reaction;
recorder Node -file "${outname}_disp.out" -node 2 -dof 1 disp;


source "D:/Git/OPS/Shared_Proc/Analysis.tcl"
set AlgOrder [list Newton NewtonLineSearch]
set DispList [list {{DispList}}];
constraints Plain;
numberer Plain;
system BandGeneral;
set isFinish [Analyse_Static_Disp_Cyclic_Control 2 1 $DispList $d_incr 1.0E-3 50 $AlgOrder]
#print -node 1;
