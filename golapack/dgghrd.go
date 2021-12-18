package golapack

import (
	"fmt"

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
// original equation A*x = lambda*B*x, then Dgghrd reduces the original
// problem to generalized Hessenberg form.
func Dgghrd(compq, compz byte, n, ilo, ihi int, a, b, q, z *mat.Matrix) (err error) {
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
	if icompq <= 0 {
		err = fmt.Errorf("icompq <= 0: compq='%c'", compq)
	} else if icompz <= 0 {
		err = fmt.Errorf("icompz <= 0: compz='%c'", compz)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if ilo < 1 {
		err = fmt.Errorf("ilo < 1: ilo=%v", ilo)
	} else if ihi > n || ihi < ilo-1 {
		err = fmt.Errorf("ihi > n || ihi < ilo-1: ilo=%v, ihi=%v", ilo, ihi)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	} else if (ilq && q.Rows < n) || q.Rows < 1 {
		err = fmt.Errorf("(ilq && q.Rows < n) || q.Rows < 1: ilq=%v, q.Rows=%v, n=%v", ilq, q.Rows, n)
	} else if (ilz && z.Rows < n) || z.Rows < 1 {
		err = fmt.Errorf("(ilz && z.Rows < n) || z.Rows < 1: ilz=%v, z.Rows=%v, n=%v", ilz, z.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Dgghrd", err)
		return
	}

	//     Initialize Q and Z if desired.
	if icompq == 3 {
		Dlaset(Full, n, n, zero, one, q)
	}
	if icompz == 3 {
		Dlaset(Full, n, n, zero, one, z)
	}

	//     Quick return if possible
	if n <= 1 {
		return
	}

	//     Zero out lower triangle of B
	for jcol = 1; jcol <= n-1; jcol++ {
		for jrow = jcol + 1; jrow <= n; jrow++ {
			b.Set(jrow-1, jcol-1, zero)
		}
	}

	//     Reduce A and B
	for jcol = ilo; jcol <= ihi-2; jcol++ {

		for jrow = ihi; jrow >= jcol+2; jrow-- {
			//           Step 1: rotate rows JROW-1, JROW to kill A(JROW,JCOL)
			temp = a.Get(jrow-1-1, jcol-1)
			c, s, *a.GetPtr(jrow-1-1, jcol-1) = Dlartg(temp, a.Get(jrow-1, jcol-1))
			a.Set(jrow-1, jcol-1, zero)
			a.Off(jrow-1, jcol).Vector().Rot(n-jcol, a.Off(jrow-1-1, jcol).Vector(), a.Rows, a.Rows, c, s)
			b.Off(jrow-1, jrow-1-1).Vector().Rot(n+2-jrow, b.Off(jrow-1-1, jrow-1-1).Vector(), b.Rows, b.Rows, c, s)
			if ilq {
				q.Off(0, jrow-1).Vector().Rot(n, q.Off(0, jrow-1-1).Vector(), 1, 1, c, s)
			}

			//           Step 2: rotate columns JROW, JROW-1 to kill B(JROW,JROW-1)
			temp = b.Get(jrow-1, jrow-1)
			c, s, *b.GetPtr(jrow-1, jrow-1) = Dlartg(temp, b.Get(jrow-1, jrow-1-1))
			b.Set(jrow-1, jrow-1-1, zero)
			a.Off(0, jrow-1-1).Vector().Rot(ihi, a.Off(0, jrow-1).Vector(), 1, 1, c, s)
			b.Off(0, jrow-1-1).Vector().Rot(jrow-1, b.Off(0, jrow-1).Vector(), 1, 1, c, s)
			if ilz {
				z.Off(0, jrow-1-1).Vector().Rot(n, z.Off(0, jrow-1).Vector(), 1, 1, c, s)
			}
		}
	}

	return
}
