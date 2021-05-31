package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgghrd reduces a pair of real matrices (A,B) to generalized upper
// Hessenberg form using orthogonal transformations, where A is a
// general matrix and B is upper triangular.  The form of the
// generalized eigenvalue problem is
//    A*x = lambda*B*x,
// and B is typically made upper triangular by computing its QR
// factorization and moving the orthogonal matrix Q to the left side
// of the equation.
//
// This subroutine simultaneously reduces A to a Hessenberg matrix H:
//    Q**T*A*Z = H
// and transforms B to another upper triangular matrix T:
//    Q**T*B*Z = T
// in order to reduce the problem to its standard form
//    H*y = lambda*T*y
// where y = Z**T*x.
//
// The orthogonal matrices Q and Z are determined as products of Givens
// rotations.  They may either be formed explicitly, or they may be
// postmultiplied into input matrices Q1 and Z1, so that
//
//      Q1 * A * Z1**T = (Q1*Q) * H * (Z1*Z)**T
//
//      Q1 * B * Z1**T = (Q1*Q) * T * (Z1*Z)**T
//
// If Q1 is the orthogonal matrix from the QR factorization of B in the
// original equation A*x = lambda*B*x, then DGGHRD reduces the original
// problem to generalized Hessenberg form.
func Dgghrd(compq, compz byte, n, ilo, ihi *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, q *mat.Matrix, ldq *int, z *mat.Matrix, ldz, info *int) {
	var ilq, ilz bool
	var c, one, s, temp, zero float64
	var icompq, icompz, jcol, jrow int

	one = 1.0
	zero = 0.0

	//     Decode COMPQ
	if compq == 'N' {
		ilq = false
		icompq = 1
	} else if compq == 'V' {
		ilq = true
		icompq = 2
	} else if compq == 'I' {
		ilq = true
		icompq = 3
	} else {
		icompq = 0
	}

	//     Decode COMPZ
	if compz == 'N' {
		ilz = false
		icompz = 1
	} else if compz == 'V' {
		ilz = true
		icompz = 2
	} else if compz == 'I' {
		ilz = true
		icompz = 3
	} else {
		icompz = 0
	}

	//     Test the input parameters.
	(*info) = 0
	if icompq <= 0 {
		(*info) = -1
	} else if icompz <= 0 {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*ilo) < 1 {
		(*info) = -4
	} else if (*ihi) > (*n) || (*ihi) < (*ilo)-1 {
		(*info) = -5
	} else if (*lda) < maxint(1, *n) {
		(*info) = -7
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -9
	} else if (ilq && (*ldq) < (*n)) || (*ldq) < 1 {
		(*info) = -11
	} else if (ilz && (*ldz) < (*n)) || (*ldz) < 1 {
		(*info) = -13
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DGGHRD"), -(*info))
		return
	}

	//     Initialize Q and Z if desired.
	if icompq == 3 {
		Dlaset('F', n, n, &zero, &one, q, ldq)
	}
	if icompz == 3 {
		Dlaset('F', n, n, &zero, &one, z, ldz)
	}

	//     Quick return if possible
	if (*n) <= 1 {
		return
	}

	//     Zero out lower triangle of B
	for jcol = 1; jcol <= (*n)-1; jcol++ {
		for jrow = jcol + 1; jrow <= (*n); jrow++ {
			b.Set(jrow-1, jcol-1, zero)
		}
	}

	//     Reduce A and B
	for jcol = (*ilo); jcol <= (*ihi)-2; jcol++ {

		for jrow = (*ihi); jrow >= jcol+2; jrow-- {
			//           Step 1: rotate rows JROW-1, JROW to kill A(JROW,JCOL)
			temp = a.Get(jrow-1-1, jcol-1)
			Dlartg(&temp, a.GetPtr(jrow-1, jcol-1), &c, &s, a.GetPtr(jrow-1-1, jcol-1))
			a.Set(jrow-1, jcol-1, zero)
			goblas.Drot(toPtr((*n)-jcol), a.Vector(jrow-1-1, jcol+1-1), lda, a.Vector(jrow-1, jcol+1-1), lda, &c, &s)
			goblas.Drot(toPtr((*n)+2-jrow), b.Vector(jrow-1-1, jrow-1-1), ldb, b.Vector(jrow-1, jrow-1-1), ldb, &c, &s)
			if ilq {
				goblas.Drot(n, q.Vector(0, jrow-1-1), func() *int { y := 1; return &y }(), q.Vector(0, jrow-1), func() *int { y := 1; return &y }(), &c, &s)
			}

			//           Step 2: rotate columns JROW, JROW-1 to kill B(JROW,JROW-1)
			temp = b.Get(jrow-1, jrow-1)
			Dlartg(&temp, b.GetPtr(jrow-1, jrow-1-1), &c, &s, b.GetPtr(jrow-1, jrow-1))
			b.Set(jrow-1, jrow-1-1, zero)
			goblas.Drot(ihi, a.Vector(0, jrow-1), func() *int { y := 1; return &y }(), a.Vector(0, jrow-1-1), func() *int { y := 1; return &y }(), &c, &s)
			goblas.Drot(toPtr(jrow-1), b.Vector(0, jrow-1), func() *int { y := 1; return &y }(), b.Vector(0, jrow-1-1), func() *int { y := 1; return &y }(), &c, &s)
			if ilz {
				goblas.Drot(n, z.Vector(0, jrow-1), func() *int { y := 1; return &y }(), z.Vector(0, jrow-1-1), func() *int { y := 1; return &y }(), &c, &s)
			}
		}
	}
}
