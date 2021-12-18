package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsptrd reduces a real symmetric matrix A stored in packed form to
// symmetric tridiagonal form T by an orthogonal similarity
// transformation: Q**T * A * Q = T.
func Dsptrd(uplo mat.MatUplo, n int, ap, d, e, tau *mat.Vector) (err error) {
	var upper bool
	var alpha, half, one, taui, zero float64
	var i, i1, i1i1, ii int

	one = 1.0
	zero = 0.0
	half = 1.0 / 2.0

	//     Test the input parameters
	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	}
	if err != nil {
		gltest.Xerbla2("Dsptrd", err)
		return
	}

	//     Quick return if possible
	if n <= 0 {
		return
	}

	if upper {
		//        Reduce the upper triangle of A.
		//        I1 is the index in AP of A(1,I+1).
		i1 = n*(n-1)/2 + 1
		for i = n - 1; i >= 1; i-- {
			//           Generate elementary reflector H(i) = I - tau * v * v**T
			//           to annihilate A(1:i-1,i+1)
			*ap.GetPtr(i1 + i - 1 - 1), taui = Dlarfg(i, ap.Get(i1+i-1-1), ap.Off(i1-1), 1)
			e.Set(i-1, ap.Get(i1+i-1-1))

			if taui != zero {
				//              Apply H(i) from both sides to A(1:i,1:i)
				ap.Set(i1+i-1-1, one)

				//              Compute  y := tau * A * v  storing y in TAU(1:i)
				if err = tau.Spmv(uplo, i, taui, ap, ap.Off(i1-1), 1, zero, 1); err != nil {
					panic(err)
				}

				//              Compute  w := y - 1/2 * tau * (y**T *v) * v
				alpha = -half * taui * ap.Off(i1-1).Dot(i, tau, 1, 1)
				tau.Axpy(i, alpha, ap.Off(i1-1), 1, 1)

				//              Apply the transformation as a rank-2 update:
				//                 A := A - v * w**T - w * v**T
				if err = ap.Spr2(uplo, i, -one, ap.Off(i1-1), 1, tau, 1); err != nil {
					panic(err)
				}

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
		for i = 1; i <= n-1; i++ {
			i1i1 = ii + n - i + 1

			//           Generate elementary reflector H(i) = I - tau * v * v**T
			//           to annihilate A(i+2:n,i)
			*ap.GetPtr(ii), taui = Dlarfg(n-i, ap.Get(ii), ap.Off(ii+2-1), 1)
			e.Set(i-1, ap.Get(ii))

			if taui != zero {
				//              Apply H(i) from both sides to A(i+1:n,i+1:n)
				ap.Set(ii, one)

				//              Compute  y := tau * A * v  storing y in TAU(i:n-1)
				if err = tau.Off(i-1).Spmv(uplo, n-i, taui, ap.Off(i1i1-1), ap.Off(ii), 1, zero, 1); err != nil {
					panic(err)
				}

				//              Compute  w := y - 1/2 * tau * (y**T *v) * v
				alpha = -half * taui * ap.Off(ii).Dot(n-i, tau.Off(i-1), 1, 1)
				tau.Off(i-1).Axpy(n-i, alpha, ap.Off(ii), 1, 1)

				//              Apply the transformation as a rank-2 update:
				//                 A := A - v * w**T - w * v**T
				if err = ap.Off(i1i1-1).Spr2(uplo, n-i, -one, ap.Off(ii), 1, tau.Off(i-1), 1); err != nil {
					panic(err)
				}

				ap.Set(ii, e.Get(i-1))
			}
			d.Set(i-1, ap.Get(ii-1))
			tau.Set(i-1, taui)
			ii = i1i1
		}
		d.Set(n-1, ap.Get(ii-1))
	}

	return
}
