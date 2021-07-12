package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsytd2 reduces a real symmetric matrix A to symmetric tridiagonal
// form T by an orthogonal similarity transformation: Q**T * A * Q = T.
func Dsytd2(uplo byte, n *int, a *mat.Matrix, lda *int, d, e, tau *mat.Vector, info *int) {
	var upper bool
	var alpha, half, one, taui, zero float64
	var i int
	var err error
	_ = err

	one = 1.0
	zero = 0.0
	half = 1.0 / 2.0

	//     Test the input parameters
	(*info) = 0
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < max(1, *n) {
		(*info) = -4
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DSYTD2"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) <= 0 {
		return
	}

	if upper {
		//        Reduce the upper triangle of A
		for i = (*n) - 1; i >= 1; i-- {
			//           Generate elementary reflector H(i) = I - tau * v * v**T
			//           to annihilate A(1:i-1,i+1)
			Dlarfg(&i, a.GetPtr(i-1, i), a.Vector(0, i), func() *int { y := 1; return &y }(), &taui)
			e.Set(i-1, a.Get(i-1, i))

			if taui != zero {
				//              Apply H(i) from both sides to A(1:i,1:i)
				a.Set(i-1, i, one)

				//              Compute  x := tau * A * v  storing x in TAU(1:i)
				err = goblas.Dsymv(mat.UploByte(uplo), i, taui, a, a.Vector(0, i, 1), zero, tau.Off(0, 1))

				//              Compute  w := x - 1/2 * tau * (x**T * v) * v
				alpha = -half * taui * goblas.Ddot(i, tau.Off(0, 1), a.Vector(0, i, 1))
				goblas.Daxpy(i, alpha, a.Vector(0, i, 1), tau.Off(0, 1))

				//              Apply the transformation as a rank-2 update:
				//                 A := A - v * w**T - w * v**T
				err = goblas.Dsyr2(mat.UploByte(uplo), i, -one, a.Vector(0, i, 1), tau.Off(0, 1), a)

				a.Set(i-1, i, e.Get(i-1))
			}
			d.Set(i, a.Get(i, i))
			tau.Set(i-1, taui)
		}
		d.Set(0, a.Get(0, 0))
	} else {
		//        Reduce the lower triangle of A
		for i = 1; i <= (*n)-1; i++ {
			//           Generate elementary reflector H(i) = I - tau * v * v**T
			//           to annihilate A(i+2:n,i)
			Dlarfg(toPtr((*n)-i), a.GetPtr(i, i-1), a.Vector(min(i+2, *n)-1, i-1), func() *int { y := 1; return &y }(), &taui)
			e.Set(i-1, a.Get(i, i-1))

			if taui != zero {
				//              Apply H(i) from both sides to A(i+1:n,i+1:n)
				a.Set(i, i-1, one)

				//              Compute  x := tau * A * v  storing y in TAU(i:n-1)
				err = goblas.Dsymv(mat.UploByte(uplo), (*n)-i, taui, a.Off(i, i), a.Vector(i, i-1, 1), zero, tau.Off(i-1, 1))

				//              Compute  w := x - 1/2 * tau * (x**T * v) * v
				alpha = -half * taui * goblas.Ddot((*n)-i, tau.Off(i-1, 1), a.Vector(i, i-1, 1))
				goblas.Daxpy((*n)-i, alpha, a.Vector(i, i-1, 1), tau.Off(i-1, 1))

				//              Apply the transformation as a rank-2 update:
				//                 A := A - v * w**T - w * v**T
				err = goblas.Dsyr2(mat.UploByte(uplo), (*n)-i, -one, a.Vector(i, i-1, 1), tau.Off(i-1, 1), a.Off(i, i))

				a.Set(i, i-1, e.Get(i-1))
			}
			d.Set(i-1, a.Get(i-1, i-1))
			tau.Set(i-1, taui)
		}
		d.Set((*n)-1, a.Get((*n)-1, (*n)-1))
	}
}
