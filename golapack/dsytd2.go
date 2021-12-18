package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsytd2 reduces a real symmetric matrix A to symmetric tridiagonal
// form T by an orthogonal similarity transformation: Q**T * A * Q = T.
func Dsytd2(uplo mat.MatUplo, n int, a *mat.Matrix, d, e, tau *mat.Vector) (err error) {
	var upper bool
	var alpha, half, one, taui, zero float64
	var i int

	one = 1.0
	zero = 0.0
	half = 1.0 / 2.0

	//     Test the input parameters
	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Dsytd2", err)
		return
	}

	//     Quick return if possible
	if n <= 0 {
		return
	}

	if upper {
		//        Reduce the upper triangle of A
		for i = n - 1; i >= 1; i-- {
			//           Generate elementary reflector H(i) = I - tau * v * v**T
			//           to annihilate A(1:i-1,i+1)
			*a.GetPtr(i-1, i), taui = Dlarfg(i, a.Get(i-1, i), a.Off(0, i).Vector(), 1)
			e.Set(i-1, a.Get(i-1, i))

			if taui != zero {
				//              Apply H(i) from both sides to A(1:i,1:i)
				a.Set(i-1, i, one)

				//              Compute  x := tau * A * v  storing x in TAU(1:i)
				if err = tau.Symv(uplo, i, taui, a, a.Off(0, i).Vector(), 1, zero, 1); err != nil {
					panic(err)
				}

				//              Compute  w := x - 1/2 * tau * (x**T * v) * v
				alpha = -half * taui * a.Off(0, i).Vector().Dot(i, tau, 1, 1)
				tau.Axpy(i, alpha, a.Off(0, i).Vector(), 1, 1)

				//              Apply the transformation as a rank-2 update:
				//                 A := A - v * w**T - w * v**T
				if err = a.Syr2(uplo, i, -one, a.Off(0, i).Vector(), 1, tau, 1); err != nil {
					panic(err)
				}

				a.Set(i-1, i, e.Get(i-1))
			}
			d.Set(i, a.Get(i, i))
			tau.Set(i-1, taui)
		}
		d.Set(0, a.Get(0, 0))
	} else {
		//        Reduce the lower triangle of A
		for i = 1; i <= n-1; i++ {
			//           Generate elementary reflector H(i) = I - tau * v * v**T
			//           to annihilate A(i+2:n,i)
			*a.GetPtr(i, i-1), taui = Dlarfg(n-i, a.Get(i, i-1), a.Off(min(i+2, n)-1, i-1).Vector(), 1)
			e.Set(i-1, a.Get(i, i-1))

			if taui != zero {
				//              Apply H(i) from both sides to A(i+1:n,i+1:n)
				a.Set(i, i-1, one)

				//              Compute  x := tau * A * v  storing y in TAU(i:n-1)
				if err = tau.Off(i-1).Symv(uplo, n-i, taui, a.Off(i, i), a.Off(i, i-1).Vector(), 1, zero, 1); err != nil {
					panic(err)
				}

				//              Compute  w := x - 1/2 * tau * (x**T * v) * v
				alpha = -half * taui * a.Off(i, i-1).Vector().Dot(n-i, tau.Off(i-1), 1, 1)
				tau.Off(i-1).Axpy(n-i, alpha, a.Off(i, i-1).Vector(), 1, 1)

				//              Apply the transformation as a rank-2 update:
				//                 A := A - v * w**T - w * v**T
				if err = a.Off(i, i).Syr2(uplo, n-i, -one, a.Off(i, i-1).Vector(), 1, tau.Off(i-1), 1); err != nil {
					panic(err)
				}

				a.Set(i, i-1, e.Get(i-1))
			}
			d.Set(i-1, a.Get(i-1, i-1))
			tau.Set(i-1, taui)
		}
		d.Set(n-1, a.Get(n-1, n-1))
	}

	return
}
