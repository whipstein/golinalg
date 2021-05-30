package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dorbdb5 orthogonalizes the column vector
//      X = [ X1 ]
//          [ X2 ]
// with respect to the columns of
//      Q = [ Q1 ] .
//          [ Q2 ]
// The columns of Q must be orthonormal.
//
// If the projection is zero according to Kahan's "twice is enough"
// criterion, then some other vector from the orthogonal complement
// is returned. This vector is chosen in an arbitrary but deterministic
// way.
func Dorbdb5(m1, m2, n *int, x1 *mat.Vector, incx1 *int, x2 *mat.Vector, incx2 *int, q1 *mat.Matrix, ldq1 *int, q2 *mat.Matrix, ldq2 *int, work *mat.Vector, lwork, info *int) {
	var one, zero float64
	var childinfo, i, j int

	one = 1.0
	zero = 0.0

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
		gltest.Xerbla([]byte("DORBDB5"), -(*info))
		return
	}

	//     Project X onto the orthogonal complement of Q
	Dorbdb6(m1, m2, n, x1, incx1, x2, incx2, q1, ldq1, q2, ldq2, work, lwork, &childinfo)

	//     If the projection is nonzero, then return
	if goblas.Dnrm2(m1, x1, incx1) != zero || goblas.Dnrm2(m2, x2, incx2) != zero {
		return
	}

	//     Project each standard basis vector e_1,...,e_M1 in turn, stopping
	//     when a nonzero projection is found
	for i = 1; i <= (*m1); i++ {
		for j = 1; j <= (*m1); j++ {
			x1.Set(j-1, zero)
		}
		x1.Set(i-1, one)
		for j = 1; j <= (*m2); j++ {
			x2.Set(j-1, zero)
		}
		Dorbdb6(m1, m2, n, x1, incx1, x2, incx2, q1, ldq1, q2, ldq2, work, lwork, &childinfo)
		if goblas.Dnrm2(m1, x1, incx1) != zero || goblas.Dnrm2(m2, x2, incx2) != zero {
			return
		}
	}

	//     Project each standard basis vector e_(M1+1),...,e_(M1+M2) in turn,
	//     stopping when a nonzero projection is found
	for i = 1; i <= (*m2); i++ {
		for j = 1; j <= (*m1); j++ {
			x1.Set(j-1, zero)
		}
		for j = 1; j <= (*m2); j++ {
			x2.Set(j-1, zero)
		}
		x2.Set(i-1, one)
		Dorbdb6(m1, m2, n, x1, incx1, x2, incx2, q1, ldq1, q2, ldq2, work, lwork, &childinfo)
		if goblas.Dnrm2(m1, x1, incx1) != zero || goblas.Dnrm2(m2, x2, incx2) != zero {
			return
		}
	}
}