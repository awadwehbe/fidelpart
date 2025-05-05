OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
gate circuitgate_392000554076047104 (p0, p1, p2) q0 {
	u3(p0, p1, p2) q0;
}
gate circuitgate_911862727651753152 (p0, p1, p2) q0, q1 {
	u3(p0, p1, p2) q0;
	cx q0, q1;
}
creg m_result[3];
creg m_result[3];
creg m_result[3];
circuitgate_911862727651753152(3.141592653589793, -1.5707963267948966, 1.5707963267948966) q[0], q[1];
circuitgate_392000554076047104(1.5707963267948966, 2.220446049250313e-16, 3.141592653589793) q[2];
measure q[0] -> m_result[0];
measure q[1] -> m_result[1];
measure q[2] -> m_result[2];
