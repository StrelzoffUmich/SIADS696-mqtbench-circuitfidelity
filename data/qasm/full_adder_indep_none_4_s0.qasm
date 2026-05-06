OPENQASM 2.0;
include "qelib1.inc";
gate gate_Carry q0,q1,q2,q3 { ccx q1,q2,q3; cx q1,q2; ccx q0,q2,q3; }
gate gate_Sum q0,q1,q2 { cx q1,q2; cx q0,q2; }
qreg q[4];
creg meas[4];
gate_Carry q[0],q[1],q[2],q[3];
cx q[1],q[2];
gate_Sum q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];