package matgen

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlaror pre- or post-multiplies an M by N matrix A by a random
// orthogonal matrix U, overwriting A.  A may optionally be initialized
// to the identity matrix before multiplying by U.  U is generated using
// the method of G.W. Stewart (SIAM J. Numer. Anal. 17, 1980, 403-409).
func Dlaror(side, init byte, m, n int, a *mat.Matrix, iseed *[]int, x *mat.Vector) (err error) {
	var factor, one, toosml, xnorm, xnorms, zero float64
	var irow, itype, ixfrm, j, jcol, kbeg, nxfrm int

	zero = 0.0
	one = 1.0
	toosml = 1.0e-20

	if n == 0 || m == 0 {
		return
	}

	if side == 'L' {
		itype = 1
	} else if side == 'R' {
		itype = 2
	} else if side == 'C' || side == 'T' {
		itype = 3
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
		gltest.Xerbla2("DLAROR", err)
		return
	}

	if itype == 1 {
		nxfrm = m
	} else {
		nxfrm = n
	}

	//     Initialize A to the identity matrix if desired
	if init == 'I' {
		golapack.Dlaset(Full, m, n, zero, one, a)
	}

	//     If no rotation possible, multiply by random +/-1
	//
	//     Compute rotation by computing Householder transformations
	//     H(2), H(3), ..., H(nhouse)
	for j = 1; j <= nxfrm; j++ {
		x.Set(j-1, zero)
	}

	for ixfrm = 2; ixfrm <= nxfrm; ixfrm++ {
		kbeg = nxfrm - ixfrm + 1

		//        Generate independent normal( 0, 1 ) random numbers
		for j = kbeg; j <= nxfrm; j++ {
			x.Set(j-1, Dlarnd(3, iseed))
		}

		//        Generate a Householder transformation from the random vector X
		xnorm = x.Off(kbeg-1).Nrm2(ixfrm, 1)
		xnorms = math.Copysign(xnorm, x.Get(kbeg-1))
		x.Set(kbeg+nxfrm-1, math.Copysign(one, -x.Get(kbeg-1)))
		factor = xnorms * (xnorms + x.Get(kbeg-1))
		if math.Abs(factor) < toosml {
			err = fmt.Errorf("Random numbers are bad!")
			gltest.Xerbla2("DLAROR", err)
			return
		} else {
			factor = one / factor
		}
		x.Set(kbeg-1, x.Get(kbeg-1)+xnorms)

		//        Apply Householder transformation to A
		if itype == 1 || itype == 3 {
			//           Apply H(k) from the left.
			if err = x.Off(2*nxfrm).Gemv(Trans, ixfrm, n, one, a.Off(kbeg-1, 0), x.Off(kbeg-1), 1, zero, 1); err != nil {
				panic(err)
			}
			if err = a.Off(kbeg-1, 0).Ger(ixfrm, n, -factor, x.Off(kbeg-1), 1, x.Off(2*nxfrm), 1); err != nil {
				panic(err)
			}

		}

		if itype == 2 || itype == 3 {
			//           Apply H(k) from the right.
			if err = x.Off(2*nxfrm).Gemv(NoTrans, m, ixfrm, one, a.Off(0, kbeg-1), x.Off(kbeg-1), 1, zero, 1); err != nil {
				panic(err)
			}
			if err = a.Off(0, kbeg-1).Ger(m, ixfrm, -factor, x.Off(2*nxfrm), 1, x.Off(kbeg-1), 1); err != nil {
				panic(err)
			}

		}
	}
	x.Set(2*nxfrm-1, math.Copysign(one, Dlarnd(3, iseed)))

	//     Scale the matrix A by D.
	if itype == 1 || itype == 3 {
		for irow = 1; irow <= m; irow++ {
			a.Off(irow-1, 0).Vector().Scal(n, x.Get(nxfrm+irow-1), a.Rows)
		}
	}

	if itype == 2 || itype == 3 {
		for jcol = 1; jcol <= n; jcol++ {
			a.Off(0, jcol-1).Vector().Scal(m, x.Get(nxfrm+jcol-1), 1)
		}
	}

	return
}
