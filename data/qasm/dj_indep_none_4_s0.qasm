OPENQASM 2.0;
include "qelib1.inc";
gate gate_Oracle q0,q1,q2,q3 { x q0; x q1; cx q0,q3; cx q1,q3; cx q2,q3; x q0; x q1; }
qreg q[4];
creg c[3];
h q[0];
h q[1];
h q[2];
u2(-pi,-pi) q[3];
gate_Oracle q[0],q[1],q[2],q[3];
h q[0];
h q[1];
h q[2];
barrier q[0],q[1],q[2],q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];