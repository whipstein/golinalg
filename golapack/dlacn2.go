package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Dlacn2 estimates the 1-norm of a square, real matrix A.
// Reverse communication is used for evaluating matrix-vector products.
func Dlacn2(n int, v, x *mat.Vector, isgn *[]int, est float64, kase int, isave *[]int) (estOut float64, kaseOut int) {
	var altsgn, estold, one, temp, two, zero float64
	var i, itmax, jlast int

	itmax = 5
	zero = 0.0
	one = 1.0
	two = 2.0
	estOut = est
	kaseOut = kase

	if kaseOut == 0 {
		for i = 1; i <= n; i++ {
			x.Set(i-1, one/float64(n))
		}
		kaseOut = 1
		(*isave)[0] = 1
		return
	}

	switch (*isave)[0] {
	case 1:
		goto label20
	case 2:
		goto label40
	case 3:
		goto label70
	case 4:
		goto label110
	case 5:
		goto label140
	}

	//     ................ ENTRY   (ISAVE( 1 ) = 1)
	//     FIRST ITERATION.  X HAS BEEN OVERWRITTEN BY A*X.
label20:
	;
	if n == 1 {
		v.Set(0, x.Get(0))
		estOut = math.Abs(v.Get(0))
		//        ... QUIT
		goto label150
	}
	estOut = x.Asum(n, 1)

	for i = 1; i <= n; i++ {
		x.Set(i-1, math.Copysign(one, x.Get(i-1)))
		(*isgn)[i-1] = int(math.Round(x.Get(i - 1)))
	}
	kaseOut = 2
	(*isave)[0] = 2
	return

	//     ................ ENTRY   (ISAVE( 1 ) = 2)
	//     FIRST ITERATION.  X HAS BEEN OVERWRITTEN BY TRANSPOSE(A)*X.
label40:
	;
	(*isave)[1] = x.Iamax(n, 1)
	(*isave)[2] = 2

	//     MAIN LOOP - ITERATIONS 2,3,...,ITMAX.
label50:
	;
	for i = 1; i <= n; i++ {
		x.Set(i-1, zero)
	}
	x.Set((*isave)[1]-1, one)
	kaseOut = 1
	(*isave)[0] = 3
	return

	//     ................ ENTRY   (ISAVE( 1 ) = 3)
	//     X HAS BEEN OVERWRITTEN BY A*X.
label70:
	;
	v.Copy(n, x, 1, 1)
	estold = estOut
	estOut = v.Asum(n, 1)
	for i = 1; i <= n; i++ {
		if int(math.Round(math.Copysign(one, x.Get(i-1)))) != (*isgn)[i-1] {
			goto label90
		}
	}
	//     REPEATED math.Copysign VECTOR DETECTED, HENCE ALGORITHM HAS CONVERGED.
	goto label120

label90:
	;
	//     TEST FOR CYCLING.
	if estOut <= estold {
		goto label120
	}

	for i = 1; i <= n; i++ {
		x.Set(i-1, math.Copysign(one, x.Get(i-1)))
		(*isgn)[i-1] = int(math.Round(x.Get(i - 1)))
	}
	kaseOut = 2
	(*isave)[0] = 4
	return

	//     ................ ENTRY   (ISAVE( 1 ) = 4)
	//     X HAS BEEN OVERWRITTEN BY TRANSPOSE(A)*X.
label110:
	;
	jlast = (*isave)[1]
	(*isave)[1] = x.Iamax(n, 1)
	if (x.Get(jlast-1) != math.Abs(x.Get((*isave)[1]-1))) && ((*isave)[2] < itmax) {
		(*isave)[2] = (*isave)[2] + 1
		goto label50
	}

	//     ITERATION COMPLETE.  FINAL STAGE.
label120:
	;
	altsgn = one
	for i = 1; i <= n; i++ {
		x.Set(i-1, altsgn*(one+float64(i-1)/float64(n-1)))
		altsgn = -altsgn
	}
	kaseOut = 1
	(*isave)[0] = 5
	return

	//     ................ ENTRY   (ISAVE( 1 ) = 5)
	//     X HAS BEEN OVERWRITTEN BY A*X.
label140:
	;
	temp = two * (x.Asum(n, 1) / float64(3*n))
	if temp > estOut {
		v.Copy(n, x, 1, 1)
		estOut = temp
	}

label150:
	;
	kaseOut = 0

	return
}
