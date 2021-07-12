package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Zlacn2 estimates the 1-norm of a square, complex matrix A.
// Reverse communication is used for evaluating matrix-vector products.
func Zlacn2(n *int, v, x *mat.CVector, est *float64, kase *int, isave *[]int) {
	var cone, czero complex128
	var absxi, altsgn, estold, one, safmin, temp, two float64
	var i, itmax, jlast int

	itmax = 5
	one = 1.0
	two = 2.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	safmin = Dlamch(SafeMinimum)
	if (*kase) == 0 {
		for i = 1; i <= (*n); i++ {
			x.SetRe(i-1, one/float64(*n))
		}
		(*kase) = 1
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
		goto label90
	case 5:
		goto label120
	}

	//     ................ ENTRY   (ISAVE( 1 ) = 1)
	//     FIRST ITERATION.  X HAS BEEN OVERWRITTEN BY A*X.
label20:
	;
	if (*n) == 1 {
		v.Set(0, x.Get(0))
		(*est) = v.GetMag(0)
		//        ... QUIT
		goto label130
	}
	(*est) = Dzsum1(n, x, func() *int { y := 1; return &y }())

	for i = 1; i <= (*n); i++ {
		absxi = x.GetMag(i - 1)
		if absxi > safmin {
			x.Set(i-1, complex(real(x.Get(i-1))/absxi, imag(x.Get(i-1))/absxi))
		} else {
			x.Set(i-1, cone)
		}
	}
	(*kase) = 2
	(*isave)[0] = 2
	return

	//     ................ ENTRY   (ISAVE( 1 ) = 2)
	//     FIRST ITERATION.  X HAS BEEN OVERWRITTEN BY CTRANS(A)*X.
label40:
	;
	(*isave)[1] = Izmax1(n, x, func() *int { y := 1; return &y }())
	(*isave)[2] = 2

	//     MAIN LOOP - ITERATIONS 2,3,...,ITMAX.
label50:
	;
	for i = 1; i <= (*n); i++ {
		x.Set(i-1, czero)
	}
	x.Set((*isave)[2-1]-1, cone)
	(*kase) = 1
	(*isave)[0] = 3
	return

	//     ................ ENTRY   (ISAVE( 1 ) = 3)
	//     X HAS BEEN OVERWRITTEN BY A*X.
label70:
	;
	goblas.Zcopy(*n, x.Off(0, 1), v.Off(0, 1))
	estold = (*est)
	(*est) = Dzsum1(n, v, func() *int { y := 1; return &y }())

	//     TEST FOR CYCLING.
	if (*est) <= estold {
		goto label100
	}

	for i = 1; i <= (*n); i++ {
		absxi = x.GetMag(i - 1)
		if absxi > safmin {
			x.Set(i-1, complex(real(x.Get(i-1))/absxi, imag(x.Get(i-1))/absxi))
		} else {
			x.Set(i-1, cone)
		}
	}
	(*kase) = 2
	(*isave)[0] = 4
	return

	//     ................ ENTRY   (ISAVE( 1 ) = 4)
	//     X HAS BEEN OVERWRITTEN BY CTRANS(A)*X.
label90:
	;
	jlast = (*isave)[1]
	(*isave)[1] = Izmax1(n, x, func() *int { y := 1; return &y }())
	if (x.GetMag(jlast-1) != x.GetMag((*isave)[2-1]-1)) && ((*isave)[2] < itmax) {
		(*isave)[2] = (*isave)[2] + 1
		goto label50
	}

	//     ITERATION COMPLETE.  FINAL STAGE.
label100:
	;
	altsgn = one
	for i = 1; i <= (*n); i++ {
		x.SetRe(i-1, altsgn*(one+float64(i-1)/float64((*n)-1)))
		altsgn = -altsgn
	}
	(*kase) = 1
	(*isave)[0] = 5
	return

	//     ................ ENTRY   (ISAVE( 1 ) = 5)
	//     X HAS BEEN OVERWRITTEN BY A*X.
label120:
	;
	temp = two * (Dzsum1(n, x, func() *int { y := 1; return &y }()) / float64(3*(*n)))
	if temp > (*est) {
		goblas.Zcopy(*n, x.Off(0, 1), v.Off(0, 1))
		(*est) = temp
	}

label130:
	;
	(*kase) = 0
}
