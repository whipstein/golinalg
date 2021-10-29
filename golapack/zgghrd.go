package golapack

import (
	"fmt"
	"math/cmplx"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgghrd reduces a pair of complex matrices (A,B) to generalized upper
// Hessenberg form using unitary transformations, where A is a
// general matrix and B is upper triangular.  The form of the
// generalized eigenvalue problem is
//    A*x = lambda*B*x,
// and B is typically made upper triangular by computing its QR
// factorization and moving the unitary matrix Q to the left side
// of the equation.
//
// This subroutine simultaneously reduces A to a Hessenberg matrix H:
//    Q**H*A*Z = H
// and transforms B to another upper triangular matrix T:
//    Q**H*B*Z = T
// in order to reduce the problem to its standard form
//    H*y = lambda*T*y
// where y = Z**H*x.
//
// The unitary matrices Q and Z are determined as products of Givens
// rotations.  They may either be formed explicitly, or they may be
// postmultiplied into input matrices Q1 and Z1, so that
//      Q1 * A * Z1**H = (Q1*Q) * H * (Z1*Z)**H
//      Q1 * B * Z1**H = (Q1*Q) * T * (Z1*Z)**H
// If Q1 is the unitary matrix from the QR factorization of B in the
// original equation A*x = lambda*B*x, then Zgghrd reduces the original
// problem to generalized Hessenberg form.
func Zgghrd(compq, compz byte, n, ilo, ihi int, a, b, q, z *mat.CMatrix) (err error) {
	var ilq, ilz bool
	var cone, ctemp, czero, s complex128
	var c float64
	var icompq, icompz, jcol, jrow int

	cone = (1.0 + 0.0*1i)
	czero = (0.0 + 0.0*1i)

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
		err = fmt.Errorf("ihi > n || ihi < ilo-1: ilo=%v, ihi=%v, n=%v", ilo, ihi, n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	} else if (ilq && q.Rows < n) || q.Rows < 1 {
		err = fmt.Errorf("(ilq && q.Rows < n) || q.Rows < 1: q.Rows=%v, n=%v, ilq=%v", q.Rows, n, ilq)
	} else if (ilz && z.Rows < n) || z.Rows < 1 {
		err = fmt.Errorf("(ilz && z.Rows < n) || z.Rows < 1: z.Rows=%v, n=%v, ilz=%v", z.Rows, n, ilz)
	}
	if err != nil {
		gltest.Xerbla2("Zgghrd", err)
		return
	}

	//     Initialize Q and Z if desired.
	if icompq == 3 {
		Zlaset(Full, n, n, czero, cone, q)
	}
	if icompz == 3 {
		Zlaset(Full, n, n, czero, cone, z)
	}

	//     Quick return if possible
	if n <= 1 {
		return
	}

	//     Zero out lower triangle of B
	for jcol = 1; jcol <= n-1; jcol++ {
		for jrow = jcol + 1; jrow <= n; jrow++ {
			b.Set(jrow-1, jcol-1, czero)
		}
	}

	//     Reduce A and B
	for jcol = ilo; jcol <= ihi-2; jcol++ {

		for jrow = ihi; jrow >= jcol+2; jrow-- { //
			//           Step 1: rotate rows JROW-1, JROW to kill A(JROW,JCOL)
			ctemp = a.Get(jrow-1-1, jcol-1)
			c, s, *a.GetPtr(jrow-1-1, jcol-1) = Zlartg(ctemp, a.Get(jrow-1, jcol-1))
			a.Set(jrow-1, jcol-1, czero)
			Zrot(n-jcol, a.CVector(jrow-1-1, jcol), a.CVector(jrow-1, jcol), c, s)
			Zrot(n+2-jrow, b.CVector(jrow-1-1, jrow-1-1), b.CVector(jrow-1, jrow-1-1), c, s)
			if ilq {
				Zrot(n, q.CVector(0, jrow-1-1, 1), q.CVector(0, jrow-1, 1), c, cmplx.Conj(s))
			}

			//           Step 2: rotate columns JROW, JROW-1 to kill B(JROW,JROW-1)
			ctemp = b.Get(jrow-1, jrow-1)
			c, s, *b.GetPtr(jrow-1, jrow-1) = Zlartg(ctemp, b.Get(jrow-1, jrow-1-1))
			b.Set(jrow-1, jrow-1-1, czero)
			Zrot(ihi, a.CVector(0, jrow-1, 1), a.CVector(0, jrow-1-1, 1), c, s)
			Zrot(jrow-1, b.CVector(0, jrow-1, 1), b.CVector(0, jrow-1-1, 1), c, s)
			if ilz {
				Zrot(n, z.CVector(0, jrow-1, 1), z.CVector(0, jrow-1-1, 1), c, s)
			}
		}
	}

	return
}
