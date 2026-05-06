OPENQASM 2.0;
include "qelib1.inc";
gate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }
gate ecr q0,q1 { s q0; sx q1; cx q0,q1; x q0; }
gate csdg q0,q1 { tdg q0; cx q0,q1; t q1; cx q0,q1; tdg q1; }
gate r(param0,param1) q0 { u(param0,-pi/2 + param1,pi/2 - param1) q0; }
gate rcccx q0,q1,q2,q3 { h q3; t q3; cx q2,q3; tdg q3; h q3; cx q0,q3; t q3; cx q1,q3; tdg q3; cx q0,q3; t q3; cx q1,q3; tdg q3; h q3; t q3; cx q2,q3; tdg q3; h q3; }
qreg q[7];
creg meas[7];
t q[3];
ccz q[4],q[2],q[0];
ecr q[3],q[0];
cswap q[1],q[5],q[6];
c3sqrtx q[2],q[4],q[1],q[5];
cswap q[0],q[3],q[1];
csdg q[4],q[5];
u1(-pi/4) q[5];
tdg q[6];
cz q[6],q[2];
cry(0.9334849134188741) q[3],q[2];
ccx q[4],q[6],q[1];
ccz q[0],q[6],q[3];
ch q[2],q[1];
ccx q[4],q[5],q[0];
cy q[1],q[0];
cswap q[6],q[2],q[3];
ccx q[3],q[4],q[2];
h q[2];
ccz q[4],q[3],q[1];
ccz q[1],q[2],q[4];
csdg q[5],q[6];
ccz q[6],q[5],q[0];
r(0.12487864535773079,2.453068833014572) q[0];
crz(5.625348551233995) q[5],q[3];
ccx q[0],q[5],q[4];
rccx q[3],q[1],q[2];
rcccx q[2],q[1],q[4],q[5];
x q[6];
cswap q[6],q[3],q[0];
sdg q[0];
rccx q[2],q[4],q[6];
rccx q[3],q[5],q[1];
rccx q[2],q[3],q[0];
ry(1.2956580914789548) q[2];
rcccx q[6],q[4],q[1],q[5];
cu3(5.698148910708643,0.7232928624012734,2.3386467136971576) q[0],q[6];
ccx q[3],q[1],q[4];
sx q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];