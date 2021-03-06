package matgen

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlarge pre- and post-multiplies a real general n by n matrix A
// with a random orthogonal matrix: A = U*D*U'.
func Dlarge(n int, a *mat.Matrix, iseed *[]int, work *mat.Vector) (err error) {
	var one, tau, wa, wb, wn, zero float64
	var i int

	zero = 0.0
	one = 1.0

	//     Test the input arguments
	if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Dlarge", err)
		return
	}

	//     pre- and post-multiply A by random orthogonal matrix
	for i = n; i >= 1; i-- {
		//        generate random reflection
		golapack.Dlarnv(3, iseed, n-i+1, work)
		wn = work.Nrm2(n-i+1, 1)
		wa = math.Copysign(wn, work.Get(0))
		if wn == zero {
			tau = zero
		} else {
			wb = work.Get(0) + wa
			work.Off(1).Scal(n-i, one/wb, 1)
			work.Set(0, one)
			tau = wb / wa
		}

		//        multiply A(i:n,1:n) by random reflection from the left
		if err = work.Off(n).Gemv(Trans, n-i+1, n, one, a.Off(i-1, 0), work, 1, zero, 1); err != nil {
			panic(err)
		}
		if err = a.Off(i-1, 0).Ger(n-i+1, n, -tau, work, 1, work.Off(n), 1); err != nil {
			panic(err)
		}

		//        multiply A(1:n,i:n) by random reflection from the right
		if err = work.Off(n).Gemv(NoTrans, n, n-i+1, one, a.Off(0, i-1), work, 1, zero, 1); err != nil {
			panic(err)
		}
		if err = a.Off(0, i-1).Ger(n, n-i+1, -tau, work.Off(n), 1, work, 1); err != nil {
			panic(err)
		}
	}

	return
}
