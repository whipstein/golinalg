package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zunbdb5 orthogonalizes the column vector
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
func Zunbdb5(m1, m2, n *int, x1 *mat.CVector, incx1 *int, x2 *mat.CVector, incx2 *int, q1 *mat.CMatrix, ldq1 *int, q2 *mat.CMatrix, ldq2 *int, work *mat.CVector, lwork, info *int) {
	var one, zero complex128
	var childinfo, i, j int

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
	} else if (*ldq1) < max(1, *m1) {
		(*info) = -9
	} else if (*ldq2) < max(1, *m2) {
		(*info) = -11
	} else if (*lwork) < (*n) {
		(*info) = -13
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZUNBDB5"), -(*info))
		return
	}

	//     Project X onto the orthogonal complement of Q
	Zunbdb6(m1, m2, n, x1, incx1, x2, incx2, q1, ldq1, q2, ldq2, work, lwork, &childinfo)

	//     If the projection is nonzero, then return
	if goblas.Dznrm2(*m1, x1.Off(0, *incx1)) != real(zero) || goblas.Dznrm2(*m2, x2.Off(0, *incx2)) != real(zero) {
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
		Zunbdb6(m1, m2, n, x1, incx1, x2, incx2, q1, ldq1, q2, ldq2, work, lwork, &childinfo)
		if goblas.Dznrm2(*m1, x1.Off(0, *incx1)) != real(zero) || goblas.Dznrm2(*m2, x2.Off(0, *incx2)) != real(zero) {
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
		Zunbdb6(m1, m2, n, x1, incx1, x2, incx2, q1, ldq1, q2, ldq2, work, lwork, &childinfo)
		if goblas.Dznrm2(*m1, x1.Off(0, *incx1)) != real(zero) || goblas.Dznrm2(*m2, x2.Off(0, *incx2)) != real(zero) {
			return
		}
	}
}
