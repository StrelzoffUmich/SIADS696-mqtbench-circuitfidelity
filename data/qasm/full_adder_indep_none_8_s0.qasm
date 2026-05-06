OPENQASM 2.0;
include "qelib1.inc";
gate gate_MAJ q0,q1,q2 { cx q0,q1; cx q0,q2; ccx q2,q1,q0; }
gate gate_UMA q0,q1,q2 { ccx q2,q1,q0; cx q0,q2; cx q2,q1; }
qreg q[8];
creg meas[8];
gate_MAJ q[1],q[4],q[0];
gate_MAJ q[2],q[5],q[1];
gate_MAJ q[3],q[6],q[2];
cx q[3],q[7];
gate_UMA q[3],q[6],q[2];
gate_UMA q[2],q[5],q[1];
gate_UMA q[1],q[4],q[0];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];