reset
unset colorbox
set palette rgb 33,13,10
set autoscale xfixmax
set autoscale xfixmin
set autoscale yfixmax
set autoscale yfixmin
set pm3d map
splot "snap_cpu.bin" bin array=62x60 format='%lf' rotate=90deg with image
