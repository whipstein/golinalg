package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zunbdb6 orthogonalizes the column vector
//      X = [ X1 ]
//          [ X2 ]
// with respect to the columns of
//      Q = [ Q1 ] .
//          [ Q2 ]
// The columns of Q must be orthonormal.
//
// If the projection is zero according to Kahan's "twice is enough"
// criterion, then the zero vector is returned.
func Zunbdb6(m1, m2, n int, x1 *mat.CVector, incx1 int, x2 *mat.CVector, incx2 int, q1, q2 *mat.CMatrix, work *mat.CVector, lwork int) (err error) {
	var negone, one, zero complex128
	var alphasq, normsq1, normsq2, realone, realzero, scl1, scl2, ssq1, ssq2 float64
	var i int

	alphasq = 0.01
	realone = 1.0
	realzero = 0.0
	negone = (-1.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test input arguments
	if m1 < 0 {
		err = fmt.Errorf("m1 < 0: m1=%v", m1)
	} else if m2 < 0 {
		err = fmt.Errorf("m2 < 0: m2=%v", m2)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if incx1 < 1 {
		err = fmt.Errorf("incx1 < 1: incx1=%v", incx1)
	} else if incx2 < 1 {
		err = fmt.Errorf("incx2 < 1: incx2=%v", incx2)
	} else if q1.Rows < max(1, m1) {
		err = fmt.Errorf("q1.Rows < max(1, m1): q1.Rows=%v, m1=%v", q1.Rows, m1)
	} else if q2.Rows < max(1, m2) {
		err = fmt.Errorf("q2.Rows < max(1, m2): q2.Rows=%v, m2=%v", q2.Rows, m2)
	} else if lwork < n {
		err = fmt.Errorf("lwork < n: lwork=%v, n=%v", lwork, n)
	}

	if err != nil {
		gltest.Xerbla2("Zunbdb6", err)
		return
	}

	//     First, project X onto the orthogonal complement of Q's column
	//     space
	scl1 = realzero
	ssq1 = realone
	scl1, ssq1 = Zlassq(m1, x1, incx1, scl1, ssq1)
	scl2 = realzero
	ssq2 = realone
	scl2, ssq2 = Zlassq(m2, x2, incx2, scl2, ssq2)
	normsq1 = math.Pow(scl1, 2)*ssq1 + math.Pow(scl2, 2)*ssq2

	if m1 == 0 {
		for i = 1; i <= n; i++ {
			work.Set(i-1, zero)
		}
	} else {
		err = work.Gemv(ConjTrans, m1, n, one, q1, x1, incx1, zero, 1)
	}

	err = work.Gemv(ConjTrans, m2, n, one, q2, x2, incx2, one, 1)

	err = x1.Gemv(NoTrans, m1, n, negone, q1, work, 1, one, incx1)
	err = x2.Gemv(NoTrans, m2, n, negone, q2, work, 1, one, incx2)

	scl1 = realzero
	ssq1 = realone
	scl1, ssq1 = Zlassq(m1, x1, incx1, scl1, ssq1)
	scl2 = realzero
	ssq2 = realone
	scl2, ssq2 = Zlassq(m2, x2, incx2, scl2, ssq2)
	normsq2 = math.Pow(scl1, 2)*ssq1 + math.Pow(scl2, 2)*ssq2

	//     If projection is sufficiently large in norm, then stop.
	//     If projection is zero, then stop.
	//     Otherwise, project again.
	if normsq2 >= alphasq*normsq1 {
		return
	}

	if complex(normsq2, 0) == zero {
		return
	}

	normsq1 = normsq2

	for i = 1; i <= n; i++ {
		work.Set(i-1, zero)
	}

	if m1 == 0 {
		for i = 1; i <= n; i++ {
			work.Set(i-1, zero)
		}
	} else {
		err = work.Gemv(ConjTrans, m1, n, one, q1, x1, incx1, zero, 1)
	}

	err = work.Gemv(ConjTrans, m2, n, one, q2, x2, incx2, one, 1)

	err = x1.Gemv(NoTrans, m1, n, negone, q1, work, 1, one, incx1)
	err = x2.Gemv(NoTrans, m2, n, negone, q2, work, 1, one, incx2)

	scl1 = realzero
	ssq1 = realone
	scl1, ssq1 = Zlassq(m1, x1, incx1, scl1, ssq1)
	scl2 = realzero
	ssq2 = realone
	scl1, ssq1 = Zlassq(m1, x1, incx1, scl1, ssq1)
	normsq2 = math.Pow(scl1, 2)*ssq1 + math.Pow(scl2, 2)*ssq2

	//     If second projection is sufficiently large in norm, then do
	//     nothing more. Alternatively, if it shrunk significantly, then
	//     truncate it to zero.
	if normsq2 < alphasq*normsq1 {
		for i = 1; i <= m1; i++ {
			x1.Set(i-1, zero)
		}
		for i = 1; i <= m2; i++ {
			x2.Set(i-1, zero)
		}
	}

	return
}
