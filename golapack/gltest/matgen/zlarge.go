package matgen

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zlarge pre- and post-multiplies a complex general n by n matrix A
// with a random unitary matrix: A = U*D*U'.
func Zlarge(n int, a *mat.CMatrix, iseed *[]int, work *mat.CVector) (err error) {
	var one, tau, wa, wb, zero complex128
	var wn float64
	var i int

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)

	//     Test the input arguments
	if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Zlarge", err)
		return
	}

	//     pre- and post-multiply A by random unitary matrix
	for i = n; i >= 1; i-- { //
		//        generate random reflection
		golapack.Zlarnv(3, iseed, n-i+1, work)
		wn = goblas.Dznrm2(n-i+1, work.Off(0, 1))
		wa = complex(wn/work.GetMag(0), 0) * work.Get(0)
		if complex(wn, 0) == zero {
			tau = zero
		} else {
			wb = work.Get(0) + wa
			goblas.Zscal(n-i, one/wb, work.Off(1, 1))
			work.Set(0, one)
			tau = complex(real(wb/wa), 0)
		}

		//        multiply A(i:n,1:n) by random reflection from the left
		if err = goblas.Zgemv(ConjTrans, n-i+1, n, one, a.Off(i-1, 0), work.Off(0, 1), zero, work.Off(n, 1)); err != nil {
			panic(err)
		}
		if err = goblas.Zgerc(n-i+1, n, -tau, work.Off(0, 1), work.Off(n, 1), a.Off(i-1, 0)); err != nil {
			panic(err)
		}

		//        multiply A(1:n,i:n) by random reflection from the right
		if err = goblas.Zgemv(NoTrans, n, n-i+1, one, a.Off(0, i-1), work.Off(0, 1), zero, work.Off(n, 1)); err != nil {
			panic(err)
		}
		if err = goblas.Zgerc(n, n-i+1, -tau, work.Off(n, 1), work.Off(0, 1), a.Off(0, i-1)); err != nil {
			panic(err)
		}
	}

	return
}
