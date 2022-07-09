# Sample Output

The program displays the following output when fed the given input / weight files:

Input file:
```
2.919385833029618, 1.6825032432987834, 2.0649861537676237, 5.402840810977821
```

Weight file:
```
4
-0.14751046652300515, -2.1604179228776754, 1.4275918461026382, 0, 5.989728207724674
-0.46318074800139586, -3.416298897283345, 0.08382614478669392, -2.551240484875474, 0
5.721459949507619, 0.45228339462709766, 1.5306335595602762, 4.84404523331882, 0
0, 5.379054906823185, 0, 0, 4.020754848023548
-3.7073439754338198, 0.1215815251738599, 2.175724591375463, -2.1091737881437567, 0
-0.1904063056175227, 0, -0.132812804396246, 0, -1.6767947695810799
-2.514967330198564, 0, 5.261210402457895, -1.1703626975513512, 2.0846516414367056
-2.5600506152834956, 5.514861210846933, -2.603943011480953, -3.3672656099984164, 2.818409700332813
0, 5.2503700304851915, 0, 0, 1.4276744918804294
5.325695682168213, 4.599905388749722, -1.0922351013272866, -0.36108430751542775, 4.450752888794689
0.07624769736844961, -3.1192292381596474, -1.8876851319759576, 1.2065289363460803, 5.032505066101791
4.786813672315811, -0.9174855013877079, 2.3887677248306414, 0.6384236389105968, 0
-0.53213522029389, -3.37564954534667, 0, -0.13427287450696745, -3.2873250969282117
-0.8575647960768893, 5.493656229815734, 0, -2.660444935637633, 0
2.212570815251045, 3.1834150447179344, -0.052726275746018736, 5.551257353101102, 0.5884530474267047
0, -1.8321921670152426, 5.671185453391683, 2.774476445819815, -2.685888370478806
0, 2.306632584628227, -2.329179475023356, 4.265333256758664, -1.5461845002240011
0, 2.7637667924454004, 0, 0, 5.808219142358963
0.172803094178823, 0.2718269368974484, -0.3836588924529858, 4.229615079839469, 3.5681949747090274
-0.23602305275211855, 0, 0
-1.2411521178310254, -3.9089328026642165, -2.7332324312858276
-0.972104248061676, -0.5510998919666319, 0
0, 0.5743954818010346, 0.8272647219023659
-3.8856876229758637, 3.6307178526515393, 0.5766469786667585
```

Will produce the following output:
```
Layer: 0 outputs: 2.919385833029618 1.6825032432987834 2.0649861537676237 5.402840810977821
Layer: 1 outputs: 10.604792498132484 17.94110887032914 7.469466278321508 5.7104159447892755 39.20982605716785
Layer: 2 outputs: -76.1361313543393 238.6475938688869 45.11913483380423 -50.33782229820534 57.56083780812283
Layer: 3 outputs: 184.81107288360045 -469.5732492394465 -342.71238166486097 163.60291039959503 1200.995225158176 
Layer: 4 outputs: 616.4432771988131 1366.9534940411459 -470.5168975152884 6105.68134242381 5344.3793220240605

Final Layer Outputs: -22608.6806323157 17567.679839645698 4396.633344686233
Largest index: 1
Largest output: 17567.679839645698
```

# Can you reuse your synchronization mechanism? Why or why not?

I used a Cyclic Barrier for my solution, which resets it's count once the the count reaches zero. Therefor I can reuse
my mechanism. I would just need to call the reset method on the Cyclic barrier instead of creating a new one
for each layer.

If I were to use a countdown latch, the countdown would never reset, therefor I could not reuse my mechanism if it were part of my solution.