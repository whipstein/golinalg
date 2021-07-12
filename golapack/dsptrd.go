package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsptrd reduces a real symmetric matrix A stored in packed form to
// symmetric tridiagonal form T by an orthogonal similarity
// transformation: Q**T * A * Q = T.
func Dsptrd(uplo byte, n *int, ap, d, e, tau *mat.Vector, info *int) {
	var upper bool
	var alpha, half, one, taui, zero float64
	var i, i1, i1i1, ii int
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
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DSPTRD"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) <= 0 {
		return
	}

	if upper {
		//        Reduce the upper triangle of A.
		//        I1 is the index in AP of A(1,I+1).
		i1 = (*n)*((*n)-1)/2 + 1
		for i = (*n) - 1; i >= 1; i-- {
			//           Generate elementary reflector H(i) = I - tau * v * v**T
			//           to annihilate A(1:i-1,i+1)
			Dlarfg(&i, ap.GetPtr(i1+i-1-1), ap.Off(i1-1), func() *int { y := 1; return &y }(), &taui)
			e.Set(i-1, ap.Get(i1+i-1-1))

			if taui != zero {
				//              Apply H(i) from both sides to A(1:i,1:i)
				ap.Set(i1+i-1-1, one)

				//              Compute  y := tau * A * v  storing y in TAU(1:i)
				err = goblas.Dspmv(mat.UploByte(uplo), i, taui, ap, ap.Off(i1-1, 1), zero, tau.Off(0, 1))

				//              Compute  w := y - 1/2 * tau * (y**T *v) * v
				alpha = -half * taui * goblas.Ddot(i, tau.Off(0, 1), ap.Off(i1-1, 1))
				goblas.Daxpy(i, alpha, ap.Off(i1-1, 1), tau.Off(0, 1))

				//              Apply the transformation as a rank-2 update:
				//                 A := A - v * w**T - w * v**T
				err = goblas.Dspr2(mat.UploByte(uplo), i, -one, ap.Off(i1-1, 1), tau.Off(0, 1), ap)

				ap.Set(i1+i-1-1, e.Get(i-1))
			}
			d.Set(i, ap.Get(i1+i-1))
			tau.Set(i-1, taui)
			i1 = i1 - i
		}
		d.Set(0, ap.Get(0))
	} else {
		//        Reduce the lower triangle of A. II is the index in AP of
		//        A(i,i) and I1I1 is the index of A(i+1,i+1).
		ii = 1
		for i = 1; i <= (*n)-1; i++ {
			i1i1 = ii + (*n) - i + 1

			//           Generate elementary reflector H(i) = I - tau * v * v**T
			//           to annihilate A(i+2:n,i)
			Dlarfg(toPtr((*n)-i), ap.GetPtr(ii), ap.Off(ii+2-1), func() *int { y := 1; return &y }(), &taui)
			e.Set(i-1, ap.Get(ii))

			if taui != zero {
				//              Apply H(i) from both sides to A(i+1:n,i+1:n)
				ap.Set(ii, one)

				//              Compute  y := tau * A * v  storing y in TAU(i:n-1)
				err = goblas.Dspmv(mat.UploByte(uplo), (*n)-i, taui, ap.Off(i1i1-1), ap.Off(ii, 1), zero, tau.Off(i-1, 1))

				//              Compute  w := y - 1/2 * tau * (y**T *v) * v
				alpha = -half * taui * goblas.Ddot((*n)-i, tau.Off(i-1, 1), ap.Off(ii, 1))
				goblas.Daxpy((*n)-i, alpha, ap.Off(ii, 1), tau.Off(i-1, 1))

				//              Apply the transformation as a rank-2 update:
				//                 A := A - v * w**T - w * v**T
				err = goblas.Dspr2(mat.UploByte(uplo), (*n)-i, -one, ap.Off(ii, 1), tau.Off(i-1, 1), ap.Off(i1i1-1))

				ap.Set(ii, e.Get(i-1))
			}
			d.Set(i-1, ap.Get(ii-1))
			tau.Set(i-1, taui)
			ii = i1i1
		}
		d.Set((*n)-1, ap.Get(ii-1))
	}
}
