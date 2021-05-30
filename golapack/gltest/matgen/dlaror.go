package matgen

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Dlaror pre- or post-multiplies an M by N matrix A by a random
// orthogonal matrix U, overwriting A.  A may optionally be initialized
// to the identity matrix before multiplying by U.  U is generated using
// the method of G.W. Stewart (SIAM J. Numer. Anal. 17, 1980, 403-409).
func Dlaror(side, init byte, m, n *int, a *mat.Matrix, lda *int, iseed *[]int, x *mat.Vector, info *int) {
	var factor, one, toosml, xnorm, xnorms, zero float64
	var irow, itype, ixfrm, j, jcol, kbeg, nxfrm int

	zero = 0.0
	one = 1.0
	toosml = 1.0e-20

	(*info) = 0
	if (*n) == 0 || (*m) == 0 {
		return
	}

	itype = 0
	if side == 'L' {
		itype = 1
	} else if side == 'R' {
		itype = 2
	} else if side == 'C' || side == 'T' {
		itype = 3
	}

	//     Check for argument errors.
	if itype == 0 {
		(*info) = -1
	} else if (*m) < 0 {
		(*info) = -3
	} else if (*n) < 0 || (itype == 3 && (*n) != (*m)) {
		(*info) = -4
	} else if (*lda) < (*m) {
		(*info) = -6
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DLAROR"), -(*info))
		return
	}

	if itype == 1 {
		nxfrm = (*m)
	} else {
		nxfrm = (*n)
	}

	//     Initialize A to the identity matrix if desired
	if init == 'I' {
		golapack.Dlaset('F', m, n, &zero, &one, a, lda)
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
			x.Set(j-1, Dlarnd(func() *int { y := 3; return &y }(), iseed))
		}

		//        Generate a Householder transformation from the random vector X
		xnorm = goblas.Dnrm2(&ixfrm, x.Off(kbeg-1), func() *int { y := 1; return &y }())
		xnorms = math.Copysign(xnorm, x.Get(kbeg-1))
		x.Set(kbeg+nxfrm-1, math.Copysign(one, -x.Get(kbeg-1)))
		factor = xnorms * (xnorms + x.Get(kbeg-1))
		if math.Abs(factor) < toosml {
			(*info) = 1
			gltest.Xerbla([]byte("DLAROR"), *info)
			return
		} else {
			factor = one / factor
		}
		x.Set(kbeg-1, x.Get(kbeg-1)+xnorms)

		//        Apply Householder transformation to A
		if itype == 1 || itype == 3 {
			//           Apply H(k) from the left.
			goblas.Dgemv(Trans, &ixfrm, n, &one, a.Off(kbeg-1, 0), lda, x.Off(kbeg-1), func() *int { y := 1; return &y }(), &zero, x.Off(2*nxfrm+1-1), func() *int { y := 1; return &y }())
			goblas.Dger(&ixfrm, n, toPtrf64(-factor), x.Off(kbeg-1), func() *int { y := 1; return &y }(), x.Off(2*nxfrm+1-1), func() *int { y := 1; return &y }(), a.Off(kbeg-1, 0), lda)

		}

		if itype == 2 || itype == 3 {
			//           Apply H(k) from the right.
			goblas.Dgemv(NoTrans, m, &ixfrm, &one, a.Off(0, kbeg-1), lda, x.Off(kbeg-1), func() *int { y := 1; return &y }(), &zero, x.Off(2*nxfrm+1-1), func() *int { y := 1; return &y }())
			goblas.Dger(m, &ixfrm, toPtrf64(-factor), x.Off(2*nxfrm+1-1), func() *int { y := 1; return &y }(), x.Off(kbeg-1), func() *int { y := 1; return &y }(), a.Off(0, kbeg-1), lda)

		}
	}

	x.Set(2*nxfrm-1, math.Copysign(one, Dlarnd(func() *int { y := 3; return &y }(), iseed)))

	//     Scale the matrix A by D.
	if itype == 1 || itype == 3 {
		for irow = 1; irow <= (*m); irow++ {
			goblas.Dscal(n, x.GetPtr(nxfrm+irow-1), a.Vector(irow-1, 0), lda)
		}
	}

	if itype == 2 || itype == 3 {
		for jcol = 1; jcol <= (*n); jcol++ {
			goblas.Dscal(m, x.GetPtr(nxfrm+jcol-1), a.Vector(0, jcol-1), func() *int { y := 1; return &y }())
		}
	}
}
