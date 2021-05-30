package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zpttrf computes the L*D*L**H factorization of a complex Hermitian
// positive definite tridiagonal matrix A.  The factorization may also
// be regarded as having the form A = U**H *D*U.
func Zpttrf(n *int, d *mat.Vector, e *mat.CVector, info *int) {
	var eii, eir, f, g, zero float64
	var i, i4 int

	zero = 0.0

	//     Test the input parameters.
	(*info) = 0
	if (*n) < 0 {
		(*info) = -1
		gltest.Xerbla([]byte("ZPTTRF"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Compute the L*D*L**H (or U**H *D*U) factorization of A.
	i4 = ((*n) - 1) % 4
	for i = 1; i <= i4; i++ {
		if d.Get(i-1) <= zero {
			(*info) = i
			return
		}
		eir = e.GetRe(i - 1)
		eii = e.GetIm(i - 1)
		f = eir / d.Get(i-1)
		g = eii / d.Get(i-1)
		e.Set(i-1, complex(f, g))
		d.Set(i+1-1, d.Get(i+1-1)-f*eir-g*eii)
	}

	for i = i4 + 1; i <= (*n)-4; i += 4 {
		//        Drop out of the loop if d(i) <= 0: the matrix is not positive
		//        definite.
		if d.Get(i-1) <= zero {
			(*info) = i
			return
		}

		//        Solve for e(i) and d(i+1).
		eir = e.GetRe(i - 1)
		eii = e.GetIm(i - 1)
		f = eir / d.Get(i-1)
		g = eii / d.Get(i-1)
		e.Set(i-1, complex(f, g))
		d.Set(i+1-1, d.Get(i+1-1)-f*eir-g*eii)

		if d.Get(i+1-1) <= zero {
			(*info) = i + 1
			return
		}

		//        Solve for e(i+1) and d(i+2).
		eir = e.GetRe(i + 1 - 1)
		eii = e.GetIm(i + 1 - 1)
		f = eir / d.Get(i+1-1)
		g = eii / d.Get(i+1-1)
		e.Set(i+1-1, complex(f, g))
		d.Set(i+2-1, d.Get(i+2-1)-f*eir-g*eii)
		//
		if d.Get(i+2-1) <= zero {
			(*info) = i + 2
			return
		}

		//        Solve for e(i+2) and d(i+3).
		eir = e.GetRe(i + 2 - 1)
		eii = e.GetIm(i + 2 - 1)
		f = eir / d.Get(i+2-1)
		g = eii / d.Get(i+2-1)
		e.Set(i+2-1, complex(f, g))
		d.Set(i+3-1, d.Get(i+3-1)-f*eir-g*eii)

		if d.Get(i+3-1) <= zero {
			(*info) = i + 3
			return
		}

		//        Solve for e(i+3) and d(i+4).
		eir = e.GetRe(i + 3 - 1)
		eii = e.GetIm(i + 3 - 1)
		f = eir / d.Get(i+3-1)
		g = eii / d.Get(i+3-1)
		e.Set(i+3-1, complex(f, g))
		d.Set(i+4-1, d.Get(i+4-1)-f*eir-g*eii)
	}

	//     Check d(n) for positive definiteness.
	if d.Get((*n)-1) <= zero {
		(*info) = (*n)
	}
}
