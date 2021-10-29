package matgen

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zlaror pre- or post-multiplies an M by N matrix A by a random
//    unitary matrix U, overwriting A. A may optionally be
//    initialized to the identity matrix before multiplying by U.
//    U is generated using the method of G.W. Stewart
//    ( SIAM J. Numer. Anal. 17, 1980, pp. 403-409 ).
//    (BLAS-2 version)
func Zlaror(side, init byte, m, n int, a *mat.CMatrix, iseed *[]int, x *mat.CVector) (err error) {
	var cone, csign, czero, xnorms complex128
	var factor, one, toosml, xabs, xnorm, zero float64
	var irow, itype, ixfrm, j, jcol, kbeg, nxfrm int

	zero = 0.0
	one = 1.0
	toosml = 1.0e-20
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	if n == 0 || m == 0 {
		return
	}

	itype = 0
	if side == 'L' {
		itype = 1
	} else if side == 'R' {
		itype = 2
	} else if side == 'C' {
		itype = 3
	} else if side == 'T' {
		itype = 4
	}

	//     Check for argument errors.
	if itype == 0 {
		err = fmt.Errorf("itype == 0: side='%c'", side)
	} else if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 || (itype == 3 && n != m) {
		err = fmt.Errorf("n < 0 || (itype == 3 && n != m): side='%c', m=%v, n=%v", side, m, n)
	} else if a.Rows < m {
		err = fmt.Errorf("a.Rows < m: a.Rows=%v, m=%v", a.Rows, m)
	}
	if err != nil {
		gltest.Xerbla2("Zlaror", err)
		return
	}

	if itype == 1 {
		nxfrm = m
	} else {
		nxfrm = n
	}

	//     Initialize A to the identity matrix if desired
	if init == 'I' {
		golapack.Zlaset(Full, m, n, czero, cone, a)
	}

	//     If no rotation possible, still multiply by
	//     a random complex number from the circle |x| = 1
	//
	//      2)      Compute Rotation by computing Householder
	//              Transformations H(2), H(3), ..., H(n).  Note that the
	//              order in which they are computed is irrelevant.
	for j = 1; j <= nxfrm; j++ {
		x.Set(j-1, czero)
	}

	for ixfrm = 2; ixfrm <= nxfrm; ixfrm++ {
		kbeg = nxfrm - ixfrm + 1

		//        Generate independent normal( 0, 1 ) random numbers
		for j = kbeg; j <= nxfrm; j++ {
			x.Set(j-1, Zlarnd(3, *iseed))
		}

		//        Generate a Householder transformation from the random vector X
		xnorm = goblas.Dznrm2(ixfrm, x.Off(kbeg-1, 1))
		xabs = x.GetMag(kbeg - 1)
		if complex(xabs, 0) != czero {
			csign = x.Get(kbeg-1) / complex(xabs, 0)
		} else {
			csign = cone
		}
		xnorms = csign * complex(xnorm, 0)
		x.Set(nxfrm+kbeg-1, -csign)
		factor = xnorm * (xnorm + xabs)
		if math.Abs(factor) < toosml {
			err = fmt.Errorf("math.Abs(factor) < toosml: factor=%v, toosml=%v", factor, toosml)
			gltest.Xerbla2("Zlaror", err)
			return
		} else {
			factor = one / factor
		}
		x.Set(kbeg-1, x.Get(kbeg-1)+xnorms)

		//        Apply Householder transformation to A
		if itype == 1 || itype == 3 || itype == 4 {
			//           Apply H(k) on the left of A
			if err = goblas.Zgemv(ConjTrans, ixfrm, n, cone, a.Off(kbeg-1, 0), x.Off(kbeg-1, 1), czero, x.Off(2*nxfrm, 1)); err != nil {
				panic(err)
			}
			if err = goblas.Zgerc(ixfrm, n, -complex(factor, 0), x.Off(kbeg-1, 1), x.Off(2*nxfrm, 1), a.Off(kbeg-1, 0)); err != nil {
				panic(err)
			}

		}

		if itype >= 2 && itype <= 4 {
			//           Apply H(k)* (or H(k)') on the right of A
			if itype == 4 {
				golapack.Zlacgv(ixfrm, x.Off(kbeg-1, 1))
			}

			if err = goblas.Zgemv(NoTrans, m, ixfrm, cone, a.Off(0, kbeg-1), x.Off(kbeg-1, 1), czero, x.Off(2*nxfrm, 1)); err != nil {
				panic(err)
			}
			if err = goblas.Zgerc(m, ixfrm, -complex(factor, 0), x.Off(2*nxfrm, 1), x.Off(kbeg-1, 1), a.Off(0, kbeg-1)); err != nil {
				panic(err)
			}

		}
	}

	x.Set(0, Zlarnd(3, *iseed))
	xabs = x.GetMag(0)
	if xabs != zero {
		csign = x.Get(0) / complex(xabs, 0)
	} else {
		csign = cone
	}
	x.Set(2*nxfrm-1, csign)

	//     Scale the matrix A by D.
	if itype == 1 || itype == 3 || itype == 4 {
		for irow = 1; irow <= m; irow++ {
			goblas.Zscal(n, x.GetConj(nxfrm+irow-1), a.CVector(irow-1, 0, *&a.Rows))
		}
	}

	if itype == 2 || itype == 3 {
		for jcol = 1; jcol <= n; jcol++ {
			goblas.Zscal(m, x.Get(nxfrm+jcol-1), a.CVector(0, jcol-1, 1))
		}
	}

	if itype == 4 {
		for jcol = 1; jcol <= n; jcol++ {
			goblas.Zscal(m, x.GetConj(nxfrm+jcol-1), a.CVector(0, jcol-1, 1))
		}
	}

	return
}
