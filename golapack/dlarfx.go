package golapack

import "github.com/whipstein/golinalg/mat"

// Dlarfx applies a real elementary reflector H to a real m by n
// matrix C, from either the left or the right. H is represented in the
// form
//
//       H = I - tau * v * v**T
//
// where tau is a real scalar and v is a real vector.
//
// If tau = 0, then H is taken to be the unit matrix
//
// This version uses inline code if H has order < 11.
func Dlarfx(side mat.MatSide, m, n int, v *mat.Vector, tau float64, c *mat.Matrix, work *mat.Vector) {
	var one, sum, t1, t10, t2, t3, t4, t5, t6, t7, t8, t9, v1, v10, v2, v3, v4, v5, v6, v7, v8, v9, zero float64
	var j int

	zero = 0.0
	one = 1.0

	if tau == zero {
		return
	}
	if side == Left {
		//        Form  H * C, where H has order m.
		switch m {
		case 1:
			goto label10
		case 2:
			goto label30
		case 3:
			goto label50
		case 4:
			goto label70
		case 5:
			goto label90
		case 6:
			goto label110
		case 7:
			goto label130
		case 8:
			goto label150
		case 9:
			goto label170
		case 10:
			goto label190
		}

		//        Code for general M
		Dlarf(side, m, n, v, 1, tau, c, work)
		return
	label10:
		;

		//        Special code for 1 x 1 Householder
		t1 = one - tau*v.Get(0)*v.Get(0)
		for j = 1; j <= n; j++ {
			c.Set(0, j-1, t1*c.Get(0, j-1))
		}
		return
	label30:
		;

		//        Special code for 2 x 2 Householder
		v1 = v.Get(0)
		t1 = tau * v1
		v2 = v.Get(1)
		t2 = tau * v2
		for j = 1; j <= n; j++ {
			sum = v1*c.Get(0, j-1) + v2*c.Get(1, j-1)
			c.Set(0, j-1, c.Get(0, j-1)-sum*t1)
			c.Set(1, j-1, c.Get(1, j-1)-sum*t2)
		}
		return
	label50:
		;

		//        Special code for 3 x 3 Householder
		v1 = v.Get(0)
		t1 = tau * v1
		v2 = v.Get(1)
		t2 = tau * v2
		v3 = v.Get(2)
		t3 = tau * v3
		for j = 1; j <= n; j++ {
			sum = v1*c.Get(0, j-1) + v2*c.Get(1, j-1) + v3*c.Get(2, j-1)
			c.Set(0, j-1, c.Get(0, j-1)-sum*t1)
			c.Set(1, j-1, c.Get(1, j-1)-sum*t2)
			c.Set(2, j-1, c.Get(2, j-1)-sum*t3)
		}
		return
	label70:
		;

		//        Special code for 4 x 4 Householder
		v1 = v.Get(0)
		t1 = tau * v1
		v2 = v.Get(1)
		t2 = tau * v2
		v3 = v.Get(2)
		t3 = tau * v3
		v4 = v.Get(3)
		t4 = tau * v4
		for j = 1; j <= n; j++ {
			sum = v1*c.Get(0, j-1) + v2*c.Get(1, j-1) + v3*c.Get(2, j-1) + v4*c.Get(3, j-1)
			c.Set(0, j-1, c.Get(0, j-1)-sum*t1)
			c.Set(1, j-1, c.Get(1, j-1)-sum*t2)
			c.Set(2, j-1, c.Get(2, j-1)-sum*t3)
			c.Set(3, j-1, c.Get(3, j-1)-sum*t4)
		}
		return
	label90:
		;

		//        Special code for 5 x 5 Householder
		v1 = v.Get(0)
		t1 = tau * v1
		v2 = v.Get(1)
		t2 = tau * v2
		v3 = v.Get(2)
		t3 = tau * v3
		v4 = v.Get(3)
		t4 = tau * v4
		v5 = v.Get(4)
		t5 = tau * v5
		for j = 1; j <= n; j++ {
			sum = v1*c.Get(0, j-1) + v2*c.Get(1, j-1) + v3*c.Get(2, j-1) + v4*c.Get(3, j-1) + v5*c.Get(4, j-1)
			c.Set(0, j-1, c.Get(0, j-1)-sum*t1)
			c.Set(1, j-1, c.Get(1, j-1)-sum*t2)
			c.Set(2, j-1, c.Get(2, j-1)-sum*t3)
			c.Set(3, j-1, c.Get(3, j-1)-sum*t4)
			c.Set(4, j-1, c.Get(4, j-1)-sum*t5)
		}
		return
	label110:
		;

		//        Special code for 6 x 6 Householder
		v1 = v.Get(0)
		t1 = tau * v1
		v2 = v.Get(1)
		t2 = tau * v2
		v3 = v.Get(2)
		t3 = tau * v3
		v4 = v.Get(3)
		t4 = tau * v4
		v5 = v.Get(4)
		t5 = tau * v5
		v6 = v.Get(5)
		t6 = tau * v6
		for j = 1; j <= n; j++ {
			sum = v1*c.Get(0, j-1) + v2*c.Get(1, j-1) + v3*c.Get(2, j-1) + v4*c.Get(3, j-1) + v5*c.Get(4, j-1) + v6*c.Get(5, j-1)
			c.Set(0, j-1, c.Get(0, j-1)-sum*t1)
			c.Set(1, j-1, c.Get(1, j-1)-sum*t2)
			c.Set(2, j-1, c.Get(2, j-1)-sum*t3)
			c.Set(3, j-1, c.Get(3, j-1)-sum*t4)
			c.Set(4, j-1, c.Get(4, j-1)-sum*t5)
			c.Set(5, j-1, c.Get(5, j-1)-sum*t6)
		}
		return
	label130:
		;

		//        Special code for 7 x 7 Householder
		v1 = v.Get(0)
		t1 = tau * v1
		v2 = v.Get(1)
		t2 = tau * v2
		v3 = v.Get(2)
		t3 = tau * v3
		v4 = v.Get(3)
		t4 = tau * v4
		v5 = v.Get(4)
		t5 = tau * v5
		v6 = v.Get(5)
		t6 = tau * v6
		v7 = v.Get(6)
		t7 = tau * v7
		for j = 1; j <= n; j++ {
			sum = v1*c.Get(0, j-1) + v2*c.Get(1, j-1) + v3*c.Get(2, j-1) + v4*c.Get(3, j-1) + v5*c.Get(4, j-1) + v6*c.Get(5, j-1) + v7*c.Get(6, j-1)
			c.Set(0, j-1, c.Get(0, j-1)-sum*t1)
			c.Set(1, j-1, c.Get(1, j-1)-sum*t2)
			c.Set(2, j-1, c.Get(2, j-1)-sum*t3)
			c.Set(3, j-1, c.Get(3, j-1)-sum*t4)
			c.Set(4, j-1, c.Get(4, j-1)-sum*t5)
			c.Set(5, j-1, c.Get(5, j-1)-sum*t6)
			c.Set(6, j-1, c.Get(6, j-1)-sum*t7)
		}
		return
	label150:
		;

		//        Special code for 8 x 8 Householder
		v1 = v.Get(0)
		t1 = tau * v1
		v2 = v.Get(1)
		t2 = tau * v2
		v3 = v.Get(2)
		t3 = tau * v3
		v4 = v.Get(3)
		t4 = tau * v4
		v5 = v.Get(4)
		t5 = tau * v5
		v6 = v.Get(5)
		t6 = tau * v6
		v7 = v.Get(6)
		t7 = tau * v7
		v8 = v.Get(7)
		t8 = tau * v8
		for j = 1; j <= n; j++ {
			sum = v1*c.Get(0, j-1) + v2*c.Get(1, j-1) + v3*c.Get(2, j-1) + v4*c.Get(3, j-1) + v5*c.Get(4, j-1) + v6*c.Get(5, j-1) + v7*c.Get(6, j-1) + v8*c.Get(7, j-1)
			c.Set(0, j-1, c.Get(0, j-1)-sum*t1)
			c.Set(1, j-1, c.Get(1, j-1)-sum*t2)
			c.Set(2, j-1, c.Get(2, j-1)-sum*t3)
			c.Set(3, j-1, c.Get(3, j-1)-sum*t4)
			c.Set(4, j-1, c.Get(4, j-1)-sum*t5)
			c.Set(5, j-1, c.Get(5, j-1)-sum*t6)
			c.Set(6, j-1, c.Get(6, j-1)-sum*t7)
			c.Set(7, j-1, c.Get(7, j-1)-sum*t8)
		}
		return
	label170:
		;

		//        Special code for 9 x 9 Householder
		v1 = v.Get(0)
		t1 = tau * v1
		v2 = v.Get(1)
		t2 = tau * v2
		v3 = v.Get(2)
		t3 = tau * v3
		v4 = v.Get(3)
		t4 = tau * v4
		v5 = v.Get(4)
		t5 = tau * v5
		v6 = v.Get(5)
		t6 = tau * v6
		v7 = v.Get(6)
		t7 = tau * v7
		v8 = v.Get(7)
		t8 = tau * v8
		v9 = v.Get(8)
		t9 = tau * v9
		for j = 1; j <= n; j++ {
			sum = v1*c.Get(0, j-1) + v2*c.Get(1, j-1) + v3*c.Get(2, j-1) + v4*c.Get(3, j-1) + v5*c.Get(4, j-1) + v6*c.Get(5, j-1) + v7*c.Get(6, j-1) + v8*c.Get(7, j-1) + v9*c.Get(8, j-1)
			c.Set(0, j-1, c.Get(0, j-1)-sum*t1)
			c.Set(1, j-1, c.Get(1, j-1)-sum*t2)
			c.Set(2, j-1, c.Get(2, j-1)-sum*t3)
			c.Set(3, j-1, c.Get(3, j-1)-sum*t4)
			c.Set(4, j-1, c.Get(4, j-1)-sum*t5)
			c.Set(5, j-1, c.Get(5, j-1)-sum*t6)
			c.Set(6, j-1, c.Get(6, j-1)-sum*t7)
			c.Set(7, j-1, c.Get(7, j-1)-sum*t8)
			c.Set(8, j-1, c.Get(8, j-1)-sum*t9)
		}
		return
	label190:
		;

		//        Special code for 10 x 10 Householder
		v1 = v.Get(0)
		t1 = tau * v1
		v2 = v.Get(1)
		t2 = tau * v2
		v3 = v.Get(2)
		t3 = tau * v3
		v4 = v.Get(3)
		t4 = tau * v4
		v5 = v.Get(4)
		t5 = tau * v5
		v6 = v.Get(5)
		t6 = tau * v6
		v7 = v.Get(6)
		t7 = tau * v7
		v8 = v.Get(7)
		t8 = tau * v8
		v9 = v.Get(8)
		t9 = tau * v9
		v10 = v.Get(9)
		t10 = tau * v10
		for j = 1; j <= n; j++ {
			sum = v1*c.Get(0, j-1) + v2*c.Get(1, j-1) + v3*c.Get(2, j-1) + v4*c.Get(3, j-1) + v5*c.Get(4, j-1) + v6*c.Get(5, j-1) + v7*c.Get(6, j-1) + v8*c.Get(7, j-1) + v9*c.Get(8, j-1) + v10*c.Get(9, j-1)
			c.Set(0, j-1, c.Get(0, j-1)-sum*t1)
			c.Set(1, j-1, c.Get(1, j-1)-sum*t2)
			c.Set(2, j-1, c.Get(2, j-1)-sum*t3)
			c.Set(3, j-1, c.Get(3, j-1)-sum*t4)
			c.Set(4, j-1, c.Get(4, j-1)-sum*t5)
			c.Set(5, j-1, c.Get(5, j-1)-sum*t6)
			c.Set(6, j-1, c.Get(6, j-1)-sum*t7)
			c.Set(7, j-1, c.Get(7, j-1)-sum*t8)
			c.Set(8, j-1, c.Get(8, j-1)-sum*t9)
			c.Set(9, j-1, c.Get(9, j-1)-sum*t10)
		}
		return
	} else {
		//        Form  C * H, where H has order n.
		switch n {
		case 1:
			goto label210
		case 2:
			goto label230
		case 3:
			goto label250
		case 4:
			goto label270
		case 5:
			goto label290
		case 6:
			goto label310
		case 7:
			goto label330
		case 8:
			goto label350
		case 9:
			goto label370
		case 10:
			goto label390
		}

		//        Code for general N
		Dlarf(side, m, n, v, 1, tau, c, work)
		return
	label210:
		;

		//        Special code for 1 x 1 Householder
		t1 = one - tau*v.Get(0)*v.Get(0)
		for j = 1; j <= m; j++ {
			c.Set(j-1, 0, t1*c.Get(j-1, 0))
		}
		return
	label230:
		;

		//        Special code for 2 x 2 Householder
		v1 = v.Get(0)
		t1 = tau * v1
		v2 = v.Get(1)
		t2 = tau * v2
		for j = 1; j <= m; j++ {
			sum = v1*c.Get(j-1, 0) + v2*c.Get(j-1, 1)
			c.Set(j-1, 0, c.Get(j-1, 0)-sum*t1)
			c.Set(j-1, 1, c.Get(j-1, 1)-sum*t2)
		}
		return
	label250:
		;

		//        Special code for 3 x 3 Householder
		v1 = v.Get(0)
		t1 = tau * v1
		v2 = v.Get(1)
		t2 = tau * v2
		v3 = v.Get(2)
		t3 = tau * v3
		for j = 1; j <= m; j++ {
			sum = v1*c.Get(j-1, 0) + v2*c.Get(j-1, 1) + v3*c.Get(j-1, 2)
			c.Set(j-1, 0, c.Get(j-1, 0)-sum*t1)
			c.Set(j-1, 1, c.Get(j-1, 1)-sum*t2)
			c.Set(j-1, 2, c.Get(j-1, 2)-sum*t3)
		}
		return
	label270:
		;

		//        Special code for 4 x 4 Householder
		v1 = v.Get(0)
		t1 = tau * v1
		v2 = v.Get(1)
		t2 = tau * v2
		v3 = v.Get(2)
		t3 = tau * v3
		v4 = v.Get(3)
		t4 = tau * v4
		for j = 1; j <= m; j++ {
			sum = v1*c.Get(j-1, 0) + v2*c.Get(j-1, 1) + v3*c.Get(j-1, 2) + v4*c.Get(j-1, 3)
			c.Set(j-1, 0, c.Get(j-1, 0)-sum*t1)
			c.Set(j-1, 1, c.Get(j-1, 1)-sum*t2)
			c.Set(j-1, 2, c.Get(j-1, 2)-sum*t3)
			c.Set(j-1, 3, c.Get(j-1, 3)-sum*t4)
		}
		return
	label290:
		;

		//        Special code for 5 x 5 Householder
		v1 = v.Get(0)
		t1 = tau * v1
		v2 = v.Get(1)
		t2 = tau * v2
		v3 = v.Get(2)
		t3 = tau * v3
		v4 = v.Get(3)
		t4 = tau * v4
		v5 = v.Get(4)
		t5 = tau * v5
		for j = 1; j <= m; j++ {
			sum = v1*c.Get(j-1, 0) + v2*c.Get(j-1, 1) + v3*c.Get(j-1, 2) + v4*c.Get(j-1, 3) + v5*c.Get(j-1, 4)
			c.Set(j-1, 0, c.Get(j-1, 0)-sum*t1)
			c.Set(j-1, 1, c.Get(j-1, 1)-sum*t2)
			c.Set(j-1, 2, c.Get(j-1, 2)-sum*t3)
			c.Set(j-1, 3, c.Get(j-1, 3)-sum*t4)
			c.Set(j-1, 4, c.Get(j-1, 4)-sum*t5)
		}
		return
	label310:
		;

		//        Special code for 6 x 6 Householder
		v1 = v.Get(0)
		t1 = tau * v1
		v2 = v.Get(1)
		t2 = tau * v2
		v3 = v.Get(2)
		t3 = tau * v3
		v4 = v.Get(3)
		t4 = tau * v4
		v5 = v.Get(4)
		t5 = tau * v5
		v6 = v.Get(5)
		t6 = tau * v6
		for j = 1; j <= m; j++ {
			sum = v1*c.Get(j-1, 0) + v2*c.Get(j-1, 1) + v3*c.Get(j-1, 2) + v4*c.Get(j-1, 3) + v5*c.Get(j-1, 4) + v6*c.Get(j-1, 5)
			c.Set(j-1, 0, c.Get(j-1, 0)-sum*t1)
			c.Set(j-1, 1, c.Get(j-1, 1)-sum*t2)
			c.Set(j-1, 2, c.Get(j-1, 2)-sum*t3)
			c.Set(j-1, 3, c.Get(j-1, 3)-sum*t4)
			c.Set(j-1, 4, c.Get(j-1, 4)-sum*t5)
			c.Set(j-1, 5, c.Get(j-1, 5)-sum*t6)
		}
		return
	label330:
		;

		//        Special code for 7 x 7 Householder
		v1 = v.Get(0)
		t1 = tau * v1
		v2 = v.Get(1)
		t2 = tau * v2
		v3 = v.Get(2)
		t3 = tau * v3
		v4 = v.Get(3)
		t4 = tau * v4
		v5 = v.Get(4)
		t5 = tau * v5
		v6 = v.Get(5)
		t6 = tau * v6
		v7 = v.Get(6)
		t7 = tau * v7
		for j = 1; j <= m; j++ {
			sum = v1*c.Get(j-1, 0) + v2*c.Get(j-1, 1) + v3*c.Get(j-1, 2) + v4*c.Get(j-1, 3) + v5*c.Get(j-1, 4) + v6*c.Get(j-1, 5) + v7*c.Get(j-1, 6)
			c.Set(j-1, 0, c.Get(j-1, 0)-sum*t1)
			c.Set(j-1, 1, c.Get(j-1, 1)-sum*t2)
			c.Set(j-1, 2, c.Get(j-1, 2)-sum*t3)
			c.Set(j-1, 3, c.Get(j-1, 3)-sum*t4)
			c.Set(j-1, 4, c.Get(j-1, 4)-sum*t5)
			c.Set(j-1, 5, c.Get(j-1, 5)-sum*t6)
			c.Set(j-1, 6, c.Get(j-1, 6)-sum*t7)
		}
		return
	label350:
		;

		//        Special code for 8 x 8 Householder
		v1 = v.Get(0)
		t1 = tau * v1
		v2 = v.Get(1)
		t2 = tau * v2
		v3 = v.Get(2)
		t3 = tau * v3
		v4 = v.Get(3)
		t4 = tau * v4
		v5 = v.Get(4)
		t5 = tau * v5
		v6 = v.Get(5)
		t6 = tau * v6
		v7 = v.Get(6)
		t7 = tau * v7
		v8 = v.Get(7)
		t8 = tau * v8
		for j = 1; j <= m; j++ {
			sum = v1*c.Get(j-1, 0) + v2*c.Get(j-1, 1) + v3*c.Get(j-1, 2) + v4*c.Get(j-1, 3) + v5*c.Get(j-1, 4) + v6*c.Get(j-1, 5) + v7*c.Get(j-1, 6) + v8*c.Get(j-1, 7)
			c.Set(j-1, 0, c.Get(j-1, 0)-sum*t1)
			c.Set(j-1, 1, c.Get(j-1, 1)-sum*t2)
			c.Set(j-1, 2, c.Get(j-1, 2)-sum*t3)
			c.Set(j-1, 3, c.Get(j-1, 3)-sum*t4)
			c.Set(j-1, 4, c.Get(j-1, 4)-sum*t5)
			c.Set(j-1, 5, c.Get(j-1, 5)-sum*t6)
			c.Set(j-1, 6, c.Get(j-1, 6)-sum*t7)
			c.Set(j-1, 7, c.Get(j-1, 7)-sum*t8)
		}
		return
	label370:
		;

		//        Special code for 9 x 9 Householder
		v1 = v.Get(0)
		t1 = tau * v1
		v2 = v.Get(1)
		t2 = tau * v2
		v3 = v.Get(2)
		t3 = tau * v3
		v4 = v.Get(3)
		t4 = tau * v4
		v5 = v.Get(4)
		t5 = tau * v5
		v6 = v.Get(5)
		t6 = tau * v6
		v7 = v.Get(6)
		t7 = tau * v7
		v8 = v.Get(7)
		t8 = tau * v8
		v9 = v.Get(8)
		t9 = tau * v9
		for j = 1; j <= m; j++ {
			sum = v1*c.Get(j-1, 0) + v2*c.Get(j-1, 1) + v3*c.Get(j-1, 2) + v4*c.Get(j-1, 3) + v5*c.Get(j-1, 4) + v6*c.Get(j-1, 5) + v7*c.Get(j-1, 6) + v8*c.Get(j-1, 7) + v9*c.Get(j-1, 8)
			c.Set(j-1, 0, c.Get(j-1, 0)-sum*t1)
			c.Set(j-1, 1, c.Get(j-1, 1)-sum*t2)
			c.Set(j-1, 2, c.Get(j-1, 2)-sum*t3)
			c.Set(j-1, 3, c.Get(j-1, 3)-sum*t4)
			c.Set(j-1, 4, c.Get(j-1, 4)-sum*t5)
			c.Set(j-1, 5, c.Get(j-1, 5)-sum*t6)
			c.Set(j-1, 6, c.Get(j-1, 6)-sum*t7)
			c.Set(j-1, 7, c.Get(j-1, 7)-sum*t8)
			c.Set(j-1, 8, c.Get(j-1, 8)-sum*t9)
		}
		return
	label390:
		;

		//        Special code for 10 x 10 Householder
		v1 = v.Get(0)
		t1 = tau * v1
		v2 = v.Get(1)
		t2 = tau * v2
		v3 = v.Get(2)
		t3 = tau * v3
		v4 = v.Get(3)
		t4 = tau * v4
		v5 = v.Get(4)
		t5 = tau * v5
		v6 = v.Get(5)
		t6 = tau * v6
		v7 = v.Get(6)
		t7 = tau * v7
		v8 = v.Get(7)
		t8 = tau * v8
		v9 = v.Get(8)
		t9 = tau * v9
		v10 = v.Get(9)
		t10 = tau * v10
		for j = 1; j <= m; j++ {
			sum = v1*c.Get(j-1, 0) + v2*c.Get(j-1, 1) + v3*c.Get(j-1, 2) + v4*c.Get(j-1, 3) + v5*c.Get(j-1, 4) + v6*c.Get(j-1, 5) + v7*c.Get(j-1, 6) + v8*c.Get(j-1, 7) + v9*c.Get(j-1, 8) + v10*c.Get(j-1, 9)
			c.Set(j-1, 0, c.Get(j-1, 0)-sum*t1)
			c.Set(j-1, 1, c.Get(j-1, 1)-sum*t2)
			c.Set(j-1, 2, c.Get(j-1, 2)-sum*t3)
			c.Set(j-1, 3, c.Get(j-1, 3)-sum*t4)
			c.Set(j-1, 4, c.Get(j-1, 4)-sum*t5)
			c.Set(j-1, 5, c.Get(j-1, 5)-sum*t6)
			c.Set(j-1, 6, c.Get(j-1, 6)-sum*t7)
			c.Set(j-1, 7, c.Get(j-1, 7)-sum*t8)
			c.Set(j-1, 8, c.Get(j-1, 8)-sum*t9)
			c.Set(j-1, 9, c.Get(j-1, 9)-sum*t10)
		}
		return
	}
}
