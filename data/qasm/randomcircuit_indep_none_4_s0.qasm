OPENQASM 2.0;
include "qelib1.inc";
gate r(param0,param1) q0 { u(param0,-pi/2 + param1,pi/2 - param1) q0; }
gate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }
qreg q[4];
creg meas[4];
cswap q[1],q[2],q[0];
t q[3];
rccx q[1],q[3],q[0];
c3sqrtx q[2],q[3],q[1],q[0];
r(2.026167409578137,1.7745821253041683) q[1];
rccx q[3],q[2],q[0];
u(3.772222268266399,1.6460880880040647,0.9388015080041642) q[0];
ccz q[3],q[2],q[1];
ccz q[2],q[1],q[3];
ccx q[3],q[1],q[0];
ch q[0],q[1];
ch q[3],q[2];
barrier q[0],q[1],q[2],q[3];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];