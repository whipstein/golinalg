package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
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
func Zunbdb6(m1, m2, n *int, x1 *mat.CVector, incx1 *int, x2 *mat.CVector, incx2 *int, q1 *mat.CMatrix, ldq1 *int, q2 *mat.CMatrix, ldq2 *int, work *mat.CVector, lwork, info *int) {
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
	(*info) = 0
	if (*m1) < 0 {
		(*info) = -1
	} else if (*m2) < 0 {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*incx1) < 1 {
		(*info) = -5
	} else if (*incx2) < 1 {
		(*info) = -7
	} else if (*ldq1) < maxint(1, *m1) {
		(*info) = -9
	} else if (*ldq2) < maxint(1, *m2) {
		(*info) = -11
	} else if (*lwork) < (*n) {
		(*info) = -13
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZUNBDB6"), -(*info))
		return
	}

	//     First, project X onto the orthogonal complement of Q's column
	//     space
	scl1 = realzero
	ssq1 = realone
	Zlassq(m1, x1, incx1, &scl1, &ssq1)
	scl2 = realzero
	ssq2 = realone
	Zlassq(m2, x2, incx2, &scl2, &ssq2)
	normsq1 = math.Pow(scl1, 2)*ssq1 + math.Pow(scl2, 2)*ssq2

	if (*m1) == 0 {
		for i = 1; i <= (*n); i++ {
			work.Set(i-1, zero)
		}
	} else {
		goblas.Zgemv(ConjTrans, m1, n, &one, q1, ldq1, x1, incx1, &zero, work, func() *int { y := 1; return &y }())
	}

	goblas.Zgemv(ConjTrans, m2, n, &one, q2, ldq2, x2, incx2, &one, work, func() *int { y := 1; return &y }())

	goblas.Zgemv(NoTrans, m1, n, &negone, q1, ldq1, work, func() *int { y := 1; return &y }(), &one, x1, incx1)
	goblas.Zgemv(NoTrans, m2, n, &negone, q2, ldq2, work, func() *int { y := 1; return &y }(), &one, x2, incx2)

	scl1 = realzero
	ssq1 = realone
	Zlassq(m1, x1, incx1, &scl1, &ssq1)
	scl2 = realzero
	ssq2 = realone
	Zlassq(m2, x2, incx2, &scl2, &ssq2)
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

	for i = 1; i <= (*n); i++ {
		work.Set(i-1, zero)
	}

	if (*m1) == 0 {
		for i = 1; i <= (*n); i++ {
			work.Set(i-1, zero)
		}
	} else {
		goblas.Zgemv(ConjTrans, m1, n, &one, q1, ldq1, x1, incx1, &zero, work, func() *int { y := 1; return &y }())
	}

	goblas.Zgemv(ConjTrans, m2, n, &one, q2, ldq2, x2, incx2, &one, work, func() *int { y := 1; return &y }())

	goblas.Zgemv(NoTrans, m1, n, &negone, q1, ldq1, work, func() *int { y := 1; return &y }(), &one, x1, incx1)
	goblas.Zgemv(NoTrans, m2, n, &negone, q2, ldq2, work, func() *int { y := 1; return &y }(), &one, x2, incx2)

	scl1 = realzero
	ssq1 = realone
	Zlassq(m1, x1, incx1, &scl1, &ssq1)
	scl2 = realzero
	ssq2 = realone
	Zlassq(m1, x1, incx1, &scl1, &ssq1)
	normsq2 = math.Pow(scl1, 2)*ssq1 + math.Pow(scl2, 2)*ssq2

	//     If second projection is sufficiently large in norm, then do
	//     nothing more. Alternatively, if it shrunk significantly, then
	//     truncate it to zero.
	if normsq2 < alphasq*normsq1 {
		for i = 1; i <= (*m1); i++ {
			x1.Set(i-1, zero)
		}
		for i = 1; i <= (*m2); i++ {
			x2.Set(i-1, zero)
		}
	}
}